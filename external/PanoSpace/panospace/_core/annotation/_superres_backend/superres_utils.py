from __future__ import annotations

import logging
import os

import anndata as ad
import cv2
import numpy as np
import scanpy as sc
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

from tqdm import tqdm
from itertools import product

from typing import Literal, Union, List

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

logger = logging.getLogger(__name__)


_IMAGE_CACHE: dict = {}


def load_rgb_image(img_path: str) -> Image.Image:
    """Open (and keep) the H&E image as a decoded RGB ``PIL.Image``.

    A whole-slide TIFF costs tens of seconds and several GB to decode, and the
    super-resolution stage needs it three times (tissue mask, spot features,
    sub-spot features).  Caching the decoded image on the module keeps that to
    a single decode; the entry is dropped by :func:`release_image`.
    """
    cached = _IMAGE_CACHE.get(img_path)
    if cached is not None:
        return cached
    img = Image.open(img_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.load()
    _IMAGE_CACHE.clear()  # only ever keep one slide in memory
    _IMAGE_CACHE[img_path] = img
    logger.info("Loaded H&E image %s (%d x %d)", img_path, img.size[0], img.size[1])
    return img


def release_image(img_path: str = None) -> None:
    """Drop the cached decoded image(s)."""
    if img_path is None:
        _IMAGE_CACHE.clear()
    else:
        _IMAGE_CACHE.pop(img_path, None)


from transformers import AutoModel
from torchvision import transforms

import hashlib, os, json


class CacheManager:
    # Bump when the on-disk cache stops meaning what it used to -- a changed
    # format, but also a changed *definition*. The key records `mask_mode` as a
    # string, so reworking the logic behind a mode name silently serves the old
    # grid unless this is bumped (v3: mask_mode='spots' reworked; v4: the seed
    # and the proportions themselves entered the key, see below).
    _CACHE_VERSION = "v4"

    def __init__(self, base_dir="~/.panospace_cache"):
        self.base_dir = os.path.expanduser(base_dir)
        os.makedirs(self.base_dir, exist_ok=True)

    def compute_cache_id(self, img_path, params: dict):
        enriched = {**params}
        # Reflect image-file identity (O(1) stat) so caches invalidate when the
        # underlying H&E image is replaced. Pixel hashing would require reading
        # the full multi-GB image and recreate the very IO bottleneck we avoid.
        try:
            enriched["_img_size"] = os.path.getsize(img_path)
            enriched["_img_mtime"] = int(os.path.getmtime(img_path))
        except OSError:
            enriched["_img_size"] = None
            enriched["_img_mtime"] = None
        enriched["_cache_version"] = self._CACHE_VERSION
        key_str = json.dumps({"img_path": img_path, **enriched}, sort_keys=True, default=str)
        return hashlib.sha1(key_str.encode()).hexdigest()[:12]

    def get_cache_path(self, img_path, params: dict):
        cache_id = self.compute_cache_id(img_path, params)
        cache_path = os.path.join(self.base_dir, cache_id)
        os.makedirs(cache_path, exist_ok=True)
        return cache_path


def _load_dinov2_model(local_path: str, pretrained_model_name: str):
    """Load DINOv2 from local path first, fall back to HuggingFace download.

    Raises RuntimeError with actionable guidance if both fail.
    """
    local_path = os.path.expanduser(local_path)
    try:
        model = AutoModel.from_pretrained(local_path, local_files_only=True)
        logger.info(f"Successfully loaded DINOv2 from local path: {local_path}")
        return model
    except Exception as exc:
        # Python clears the `except ... as` name when the block ends, so keep a
        # copy: the message below (upstream) referenced it and raised NameError.
        local_error = exc
        logger.warning(f"Failed to load DINOv2 locally: {local_error}")

    try:
        logger.info(f"Attempting to download DINOv2 from Hugging Face: {pretrained_model_name}")
        model = AutoModel.from_pretrained(pretrained_model_name)
        logger.info("Successfully downloaded DINOv2 from Hugging Face")
        # Persist next to the other caches so later runs -- in particular on
        # compute nodes without outbound network -- never need the Hub again.
        try:
            os.makedirs(local_path, exist_ok=True)
            model.save_pretrained(local_path)
            logger.info("Cached DINOv2 weights to %s", local_path)
        except Exception as e_save:
            logger.warning("Could not cache DINOv2 weights to %s: %s", local_path, e_save)
        return model
    except Exception as e_remote:
        raise RuntimeError(
            f"Failed to load DINOv2 both locally and online.\n"
            f"Local path tried: {local_path}\n"
            f"Hugging Face model: {pretrained_model_name}\n"
            f"Error details:\nLocal load error: {local_error}\nRemote download error: {e_remote}\n\n"
            "Please check your internet connection or manually download the model from:\n"
            "https://huggingface.co/facebook/dinov2-base\n"
            "and place it in the specified local path."
        )


class FeatureExtractor:
    """One-shot extraction of frozen DINOv2 cls-token features.

    Compared to per-epoch ViT forward, this:
    - loads the H&E image once in a single process (no DataLoader worker copies);
    - batches center + neighbor crops into [2B, 3, 518, 518] so ViT runs once
      per pair instead of twice;
    - uses ``inference_mode`` + ``autocast`` + ``cudnn.benchmark`` for max throughput;
    - returns fp32 CPU tensors ready for ``torch.save`` / direct training.
    """

    def __init__(
        self,
        local_path,
        pretrained_model_name,
        device: Union[str, torch.device] = "cuda",
        precision: Literal["16-mixed", "32"] = "16-mixed",
    ):
        torch.backends.cudnn.benchmark = True  # fixed 518x518 input → autotune conv algos
        self.device = torch.device(device)
        self.vit = _load_dinov2_model(local_path, pretrained_model_name).eval().to(self.device)
        for p in self.vit.parameters():
            p.requires_grad = False
        # AMP only kicks in on CUDA; on CPU we stay fp32 to keep semantics simple.
        self.use_amp = precision == "16-mixed" and self.device.type == "cuda"
        self.amp_dtype = torch.float16 if self.use_amp else torch.float32
        self.tf = transforms.Compose(
            [
                transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(518),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @torch.inference_mode()
    def extract_all(self, centers, img_path, radius: int, neighb: int, batch_size: int = 64, desc: str = "extract"):
        """Extract (center_feat, neighbor_feat) for every (x, y) in ``centers``.

        Returns
        -------
        center_feat : Tensor[N, 768]   fp32, CPU
        neighbor_feat : Tensor[N, 768] fp32, CPU
        """
        centers = np.asarray(centers)
        n = centers.shape[0]
        if n == 0:
            empty = torch.empty(0, 768, dtype=torch.float32)
            return empty, empty

        img = load_rgb_image(img_path)
        r, k = int(radius), int(neighb)
        center_feats = torch.empty(n, 768, dtype=torch.float32)
        neighbor_feats = torch.empty(n, 768, dtype=torch.float32)

        for start in tqdm(range(0, n, batch_size), desc=desc):
            end = min(start + batch_size, n)
            batch_centers = centers[start:end]
            B = batch_centers.shape[0]

            # CPU work first: build [2B, 3, 518, 518] so center & neighbor share
            # one ViT forward.
            crops = [self.tf(img.crop((int(x) - r, int(y) - r, int(x) + r, int(y) + r))) for x, y in batch_centers]
            crops += [
                self.tf(img.crop((int(x) - r * k, int(y) - r * k, int(x) + r * k, int(y) + r * k)))
                for x, y in batch_centers
            ]
            batch = torch.stack(crops, dim=0).to(self.device, non_blocking=True)

            with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                feats = self.vit(batch).pooler_output.to(torch.float32)  # [2B, 768]

            center_feats[start:end] = feats[:B].cpu()
            neighbor_feats[start:end] = feats[B:].cpu()

        return center_feats, neighbor_feats


class CachedFeatureDataset(Dataset):
    """In-memory dataset of pre-extracted DINOv2 features (and optional labels).

    Pure tensor indexing — no PIL, no ViT. Designed for ``num_workers=0``:
    multiprocessing workers would only add IPC overhead for ~MB of features.
    """

    def __init__(self, center_feat: torch.Tensor, neighbor_feat: torch.Tensor, labels: torch.Tensor = None):
        if center_feat.shape[0] != neighbor_feat.shape[0]:
            raise ValueError(
                f"center_feat and neighbor_feat must share dim 0, got "
                f"{center_feat.shape[0]} vs {neighbor_feat.shape[0]}"
            )
        if labels is not None and labels.shape[0] != center_feat.shape[0]:
            raise ValueError(f"labels length {labels.shape[0]} != features length {center_feat.shape[0]}")
        self.center_feat = center_feat
        self.neighbor_feat = neighbor_feat
        self.labels = labels

    def __getitem__(self, i):
        if self.labels is not None:
            return self.center_feat[i], self.neighbor_feat[i], self.labels[i]
        return self.center_feat[i], self.neighbor_feat[i]

    def __len__(self):
        return self.center_feat.shape[0]


class DINOv2NeighborDataset(Dataset):
    def __init__(self, centers, img_path, label_frame=None, train=True, radius=129, neighb=3):
        self.centers = centers
        self.label_frame = label_frame
        self.train = train
        self.radius = radius
        self.neighb = neighb

        self.image = Image.open(img_path).convert("RGB")
        self.image.load()

        self.transform = ImageTransform(resize=518, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, index):
        x, y = self.centers[index]
        r, n = self.radius, self.neighb

        crop = self.image.crop((x - r, y - r, x + r, y + r))
        crop_neighbor = self.image.crop((x - r * n, y - r * n, x + r * n, y + r * n))

        if self.train:
            crop = self.transform(img=crop, phase="valid")
            crop_neighbor = self.transform(img=crop_neighbor, phase="valid")
            label = self.label_frame.iloc[index, :].values.astype(np.float32)
            return crop, crop_neighbor, label
        else:
            crop = self.transform(img=crop, phase="valid")
            crop_neighbor = self.transform(img=crop_neighbor, phase="valid")
            return crop, crop_neighbor

    def __len__(self):
        return len(self.centers)


class DINOv2NeighborClassifier(pl.LightningModule):
    """MLP head over pre-extracted DINOv2 ``(center, neighbor)`` cls-token features.

    The ViT backbone is frozen and is *not* held by this module: features are
    pre-extracted once by :class:`FeatureExtractor` and cached on disk. The
    forward pass therefore consumes feature tensors, not images.

    Legacy checkpoints (image-input based, with ``vit.*`` weights in their
    state_dict) are not compatible; see ``DINOv2_superres_deconv.__init__``
    for the legacy-detection guard.
    """

    def __init__(
        self,
        num_classes=9,
        class_weights=None,
        learning_rate=1e-4,
        feature_dropout: float = 0.1,
        noise_std: float = 0.0,
        local_path: str = "~/.panospace_cache/dinov2-base",
        pretrained_model_name: str = "facebook/dinov2-base",
    ):
        super().__init__()
        self.learning_rate = learning_rate
        # Persist constructor args so load_from_checkpoint can rebuild the head
        # without needing the DINOv2 backbone (we don't load it at all here).
        self.save_hyperparameters(ignore=["class_weights"])
        self.feature_dropout_p = float(feature_dropout)
        self.noise_std = float(noise_std)

        self.classifier = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )
        self.feat_dropout = nn.Dropout(self.feature_dropout_p)

        # Registered buffer migrates with .to(device) automatically — no more
        # per-step .to(self.device) like the legacy training_step did.
        # persistent=False keeps it out of the checkpoint; callers pass weights
        # explicitly when constructing, and load_from_checkpoint can re-supply.
        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.tensor(class_weights, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.class_weights = None

    def forward(self, center_feat, neighbor_feat):
        # Feature-space regularization. The legacy ImageTransform defined eight
        # augmentations but the train branch always passed phase='valid', so
        # they were effectively dead code. We replace that with light feature
        # noise + dropout — same regularizing role, no extra ViT cost.
        if self.training and self.noise_std > 0:
            center_feat = center_feat + torch.randn_like(center_feat) * self.noise_std
            neighbor_feat = neighbor_feat + torch.randn_like(neighbor_feat) * self.noise_std
        center_feat = self.feat_dropout(center_feat)
        neighbor_feat = self.feat_dropout(neighbor_feat)
        x = torch.cat([center_feat, neighbor_feat], dim=1)  # [B, 1536]
        return self.classifier(x)  # logits [B, num_classes]

    def _soft_cross_entropy(self, logits, soft_label):
        # Per-class soft-label CE = -sum_c label_c * log_softmax(logits)_c.
        # Numerically equivalent to the legacy KL(softmax(pred) || label) up to
        # a constant, but skips the redundant softmax → log round-trip.
        log_prob = F.log_softmax(logits, dim=1)
        per_class = -(soft_label * log_prob)  # [B, C]
        if self.class_weights is not None:
            per_class = per_class * self.class_weights  # broadcast [C] over [B, C]
        return per_class.sum(dim=1)  # [B]

    def training_step(self, batch, batch_idx):
        center_feat, neighbor_feat, label = batch
        logits = self(center_feat, neighbor_feat)
        loss = self._soft_cross_entropy(logits, label).mean()
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        center_feat, neighbor_feat, label = batch
        logits = self(center_feat, neighbor_feat)
        loss = self._soft_cross_entropy(logits, label).mean()
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = Adam(self.classifier.parameters(), lr=self.learning_rate)
        # self.trainer is attached by the time this runs; guard the test path
        # where the module is constructed standalone.
        try:
            max_epochs = self.trainer.max_epochs
        except RuntimeError:
            max_epochs = None
        if not max_epochs or max_epochs <= 0:
            max_epochs = 50
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=max_epochs,
            eta_min=self.learning_rate * 0.01,
        )
        return {"optimizer": opt, "lr_scheduler": sched}


def _fingerprint_frame(values, index) -> str:
    """Stable short hash of a label matrix and its row order."""
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(np.asarray(values, dtype=np.float64)).tobytes())
    h.update("\x00".join(map(str, index)).encode())
    return h.hexdigest()[:16]


class DINOv2_superres_deconv(object):
    def _proportions_fingerprint(self) -> str:
        return _fingerprint_frame(
            self.deconv_adata.obs[self.cell_type_name].to_numpy(),
            list(self.deconv_adata.obs_names),
        )

    def __init__(
        self,
        deconv_adata,
        img_dir,
        radius=129,
        neighb=2,
        class_weights=None,
        learning_rate=1e-4,
        local_path="~/.panospace_cache/dinov2-base",
        pretrained_model_name="facebook/dinov2-base",
        cache_dir="~/.panospace_cache",
        mask_mode="largest",
        mask_downscale=1,
        mask_min_spots=1,
        seed=42,
    ):

        self.img_dir = img_dir
        self.deconv_adata = deconv_adata
        self.cell_type_name = list(self.deconv_adata.uns["celltype"])
        num_classes = len(self.cell_type_name)
        self.mask_mode = mask_mode
        self.mask_downscale = mask_downscale
        self.mask_min_spots = int(mask_min_spots)
        self._seed = int(seed)

        params = {
            "radius": radius,
            "neighb": neighb,
            "num_classes": num_classes,
            "celltypes": self.cell_type_name,
            "dinov2_model": pretrained_model_name,
            # the sub-spot lattice depends on the mask and on the spot pitch,
            # so both belong in the cache identity
            "grid_pitch": int(round(float(deconv_adata.uns.get("radius", radius)))),
            "mask_mode": mask_mode,
            "mask_downscale": mask_downscale,
            "mask_min_spots": int(mask_min_spots),
        }
        cache_manager = CacheManager(base_dir=cache_dir)
        self.path = cache_manager.get_cache_path(img_dir, params)

        self.num_classes = num_classes
        self.radius = radius
        self.neighb = neighb
        self.local_path = local_path
        self.pretrained_model_name = pretrained_model_name
        self._features = None  # lazy: populated by .features on first access

        # Two cache identities, because two different things are stored here.
        # `self.path` holds features.pt: DINOv2 embeddings of the spots and
        # sub-spots, which depend only on the image and the lattice, so every
        # level, proportions table and seed of a slide share them.
        # `self._ckpt_dir` holds the *trained* head and its sr_adata, which do
        # depend on what they were trained on and with. Keying them together
        # was wrong in both directions: every seed reused the first seed's
        # checkpoint, so seeds varied nothing; and two proportion tables sharing
        # a cell-type list collided, so an EnDecon run could silently reuse the
        # head trained on ground-truth proportions (6 of 27 runs here).
        self._ckpt_dir = os.path.join(self.path, f"head_seed{int(seed)}_{self._proportions_fingerprint()}")
        os.makedirs(self._ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(self._ckpt_dir, "superres_model.ckpt")
        if os.path.exists(ckpt_path):
            self._validate_legacy_ckpt(ckpt_path)
            logger.info("Checkpoint exists in %s, loading from checkpoint...", self._ckpt_dir)
            logger.info("If using checkpoint, run_train method will be skipped.")
            logger.info("If you want to retrain, please delete the checkpoint file first.")
            self.train = False
            self.model = DINOv2NeighborClassifier.load_from_checkpoint(
                ckpt_path,
                num_classes=num_classes,
                class_weights=class_weights,
            )
        else:
            logger.info("Initializing new model...")
            self.train = True
            self.model = DINOv2NeighborClassifier(
                num_classes=num_classes,
                class_weights=class_weights,
                learning_rate=learning_rate,
                local_path=local_path,
                pretrained_model_name=pretrained_model_name,
            )
        logger.info("Model loaded...")
        logger.info("Loading super-res data")
        if not os.path.exists(os.path.join(self._ckpt_dir, "sr_adata.h5ad")):
            self.sr_adata = self.make_sr_datalist()
            self.sr_adata.write(os.path.join(self._ckpt_dir, "sr_adata.h5ad"))
        else:
            self.sr_adata = sc.read(os.path.join(self._ckpt_dir, "sr_adata.h5ad"))

    @staticmethod
    def _validate_legacy_ckpt(ckpt_path: str):
        """Detect legacy image-input checkpoints and refuse to load them.

        Legacy ``superres_model.ckpt`` stored ``vit.*`` weights (the frozen
        DINOv2 backbone) and used a forward(crop, crop_neighbor) signature.
        The new feature-based model holds no ViT and uses forward(cf, nf), so
        the two are structurally incompatible. Raise with actionable guidance.
        """
        try:
            snapshot = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception:
            return  # let load_from_checkpoint raise the real error downstream
        state_dict = snapshot.get("state_dict", {}) if isinstance(snapshot, dict) else {}
        has_vit = any(k.startswith("vit.") for k in state_dict)
        if has_vit:
            raise RuntimeError(
                f"Detected legacy image-input checkpoint at {ckpt_path}.\n"
                f"The super-res trainer has been refactored to use pre-extracted "
                f"DINOv2 features and is no longer compatible with this checkpoint.\n"
                f"To retrain: delete the file and re-run.\n"
                f"  rm {ckpt_path!r}"
            )

    @property
    def features(self):
        """Lazy load (or compute & cache) DINOv2 features for spots + subspots."""
        if self._features is None:
            self._features = self._precompute_features()
        return self._features

    def _precompute_features(
        self, device="cuda", precision: Literal["16-mixed", "32"] = "16-mixed", batch_size: int = 64
    ):
        """Extract & cache DINOv2 features for both spot and subspot coordinates.

        Spot features + soft labels train the MLP; subspot features drive
        inference. Cached as ``features.pt`` so subsequent runs skip the ViT.
        """
        feat_path = os.path.join(self.path, "features.pt")
        if os.path.exists(feat_path):
            bundle = torch.load(feat_path, map_location="cpu", weights_only=False)
            meta = bundle.get("meta", {}) if isinstance(bundle, dict) else {}
            if (
                meta.get("radius") == self.radius
                and meta.get("neighb") == self.neighb
                and meta.get("num_classes") == self.num_classes
                and meta.get("dinov2_model") == self.pretrained_model_name
                and meta.get("feat_version") == CacheManager._CACHE_VERSION
            ):
                logger.info("Loaded cached features from %s", feat_path)
                return bundle
            logger.warning("Cached features param mismatch — re-extracting")

        # Soft labels: clipped + row-normalized deconv proportions per spot
        deconv = np.asarray(self.deconv_adata.obs[self.cell_type_name], dtype=np.float32)
        deconv = np.clip(deconv, 0, None)
        row_sum = deconv.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        spot_labels = torch.from_numpy(deconv / row_sum)  # [N_spot, C]

        extractor = FeatureExtractor(
            local_path=self.local_path,
            pretrained_model_name=self.pretrained_model_name,
            device=device,
            precision=precision,
        )
        spot_centers = np.asarray(self.deconv_adata.obsm["spatial"])
        spot_cf, spot_nf = extractor.extract_all(
            spot_centers,
            self.img_dir,
            radius=self.radius,
            neighb=self.neighb,
            batch_size=batch_size,
            desc="spot-features",
        )
        sr_centers = np.asarray(self.sr_adata.obsm["spatial"])
        sr_cf, sr_nf = extractor.extract_all(
            sr_centers,
            self.img_dir,
            radius=self.radius,
            neighb=self.neighb,
            batch_size=batch_size,
            desc="subspot-features",
        )
        # Free ViT memory before MLP training (the head doesn't need it).
        del extractor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        bundle = {
            "spot_center": spot_cf,
            "spot_neighbor": spot_nf,
            "spot_labels": spot_labels,
            "sr_center": sr_cf,
            "sr_neighbor": sr_nf,
            "meta": {
                "radius": self.radius,
                "neighb": self.neighb,
                "num_classes": self.num_classes,
                "celltypes": self.cell_type_name,
                "dinov2_model": self.pretrained_model_name,
                "feat_version": CacheManager._CACHE_VERSION,
                "img_size": os.path.getsize(self.img_dir),
                "img_mtime": int(os.path.getmtime(self.img_dir)),
            },
        }
        torch.save(bundle, feat_path)
        logger.info("Saved features to %s (spot=%d, subspot=%d)", feat_path, spot_cf.shape[0], sr_cf.shape[0])
        return bundle

    def make_sr_datalist(self):
        """Build the sub-spot lattice: a regular grid, clipped to the tissue."""
        # `radius` doubles as the grid pitch, and `range` needs an int step.
        r = int(round(float(self.deconv_adata.uns["radius"])))
        if r < 1:
            raise ValueError(f"uns['radius'] resolved to {r} px, which is not a usable grid pitch.")
        spot_centers = np.asarray(self.deconv_adata.obsm["spatial"])

        # Build higher resolution grid coordinates (subspots)
        axis_x = range(int(spot_centers[:, 0].min()), int(spot_centers[:, 0].max()), r)
        axis_y = range(int(spot_centers[:, 1].min()), int(spot_centers[:, 1].max()), r)
        subspot_centers = np.array([*product(axis_x, axis_y)])

        mask = self._tissue_mask()

        # Filter subspots: remove points outside mask and beyond image boundaries
        x = subspot_centers[:, 0]
        y = subspot_centers[:, 1]
        inside = (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
        keep = np.zeros(subspot_centers.shape[0], dtype=bool)
        keep[inside] = mask[y[inside], x[inside]]
        subspot_centers = subspot_centers[keep]
        logger.info(
            "sub-spot grid: %d points on a %d px lattice (%d dropped outside the tissue mask)",
            subspot_centers.shape[0],
            r,
            int((~keep).sum()),
        )
        if subspot_centers.shape[0] == 0:
            raise RuntimeError(
                "The sub-spot grid is empty: the tissue mask rejected every grid point. "
                "Check that the H&E image and the spot coordinates share the same "
                "pixel space, or pass mask_mode='none'."
            )

        # Build high-res AnnData (expression matrix initialized to zeros)
        sr_adata = ad.AnnData(np.zeros((subspot_centers.shape[0], 1)))
        sr_adata.obsm["spatial"] = subspot_centers

        return sr_adata

    def _tissue_mask(self) -> np.ndarray:
        """Boolean tissue mask, at full image resolution.

        Upstream ran Canny + ``findContours`` on the full-resolution slide and
        kept the single largest contour.  On a whole slide that is ~10^9 pixels
        of edge detection for a decision taken on a `radius`-pixel lattice, so
        by default the detection runs on a downscaled proxy and the contour is
        scaled back up (``mask_downscale='auto'``).  ``mask_mode`` picks which
        contours are kept: ``'largest'`` (upstream), ``'spots'``, ``'none'``.
        """
        if self.mask_mode == "none":
            img = load_rgb_image(self.img_dir)
            return np.ones((img.size[1], img.size[0]), dtype=bool)

        img_pil = load_rgb_image(self.img_dir)
        W, H = img_pil.size

        factor = self.mask_downscale
        if factor == "auto":
            # Keep the proxy under ~16 MP; the tissue outline is a coarse object
            # and the grid pitch (>= tens of px) dwarfs the resulting error.
            factor = max(1, int(np.ceil(np.sqrt((W * H) / 16e6))))
        factor = max(1, int(factor))

        if factor > 1:
            small = img_pil.resize((max(1, W // factor), max(1, H // factor)), Image.BILINEAR)
            logger.info(
                "Tissue mask: detecting contours on a 1/%d proxy (%d x %d)", factor, small.size[0], small.size[1]
            )
        else:
            small = img_pil
        arr = np.asarray(small)

        cnt_info = cv2_detect_contour(arr, all_cnt_info=True)
        if self.mask_mode == "largest":
            contours = [cnt_info[0][0]]
        elif self.mask_mode == "spots":
            # Keep a fragment only if at least one *measured* spot sits on it.
            # Area alone cannot tell a second tissue piece from a speck of dirt;
            # the spot layout can. A fragment with no spot carries no
            # deconvolution signal to propagate anyway.
            #
            # Two filters are needed, not one. Canny also outlines texture
            # *inside* the tissue, and those inner contours contain spots too --
            # keeping them would tile the section with small patches instead of
            # covering it. So contours are walked largest-first and one is
            # skipped when it already lies inside a fragment that was kept.
            spots = np.asarray(self.deconv_adata.obsm["spatial"], dtype=np.float64) / float(factor)
            mask_small = np.zeros(arr.shape[:2], dtype=np.uint8)
            contours = []
            for c, _convex, _area in cnt_info:  # cnt_info is sorted by area, desc
                bx, by, bw, bh = cv2.boundingRect(c)
                inbox = spots[
                    (spots[:, 0] >= bx) & (spots[:, 0] < bx + bw) & (spots[:, 1] >= by) & (spots[:, 1] < by + bh)
                ]
                if inbox.shape[0] == 0:
                    continue
                nsp = sum(1 for px, py in inbox if cv2.pointPolygonTest(c, (float(px), float(py)), False) >= 0)
                if nsp < self.mask_min_spots:
                    continue
                pts = c.reshape(-1, 2)
                probe = pts[np.linspace(0, len(pts) - 1, min(len(pts), 8)).astype(int)]
                if contours and mask_small[probe[:, 1], probe[:, 0]].all():
                    continue  # nested inside a fragment already kept
                cv2.drawContours(mask_small, [c], contourIdx=-1, color=255, thickness=-1)
                contours.append(c)
            if not contours:
                raise RuntimeError(
                    f"mask_mode='spots' kept no contour: no detected tissue outline "
                    f"contains at least {self.mask_min_spots} spot centre(s). Check that "
                    f"the H&E and the spot coordinates share the same pixel space, or "
                    f"lower mask_min_spots."
                )
            logger.info(
                "Tissue mask: mode=spots, %d fragment(s) carry >= %d measured spot(s)",
                len(contours),
                self.mask_min_spots,
            )
            if factor > 1:
                return cv2.resize(mask_small, (W, H), interpolation=cv2.INTER_NEAREST) > 0
            return mask_small > 0
        else:
            raise ValueError(f"Unknown mask_mode {self.mask_mode!r}")
        logger.info("Tissue mask: mode=%s, %d contour(s) kept", self.mask_mode, len(contours))

        mask_small = np.zeros(arr.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask_small, contours, contourIdx=-1, color=255, thickness=-1)
        if factor > 1:
            mask = cv2.resize(mask_small, (W, H), interpolation=cv2.INTER_NEAREST)
        else:
            mask = mask_small
        return mask > 0

    def _predict(self, dataloader, device):
        """Run MLP forward over cached features; collect probs in one concat.

        Uses ``inference_mode`` (cheaper than ``no_grad``) and defers the
        single D2H copy + numpy materialization until the end, instead of the
        legacy per-batch ``.cpu().detach().squeeze(1).numpy()`` sync.
        """
        self.model.eval()
        model = self.model.to(device)
        outs = []
        with torch.inference_mode():
            for cf, nf in tqdm(dataloader, desc="superres-predict"):
                logits = model(cf.to(device, non_blocking=True), nf.to(device, non_blocking=True))
                outs.append(F.softmax(logits, dim=1).cpu())
        return torch.cat(outs, dim=0).numpy()

    def run_train(
        self,
        epoch=50,
        batch_size=32,
        num_workers=0,
        accelerator="gpu",
        precision="16-mixed",
        devices: Union[int, str] = "auto",
        seed: int = 42,
        feature_batch_size: int = 64,
        extract_device: Union[str, None] = None,
        extract_precision: Union[str, None] = None,
        val_frac: float = 0.15,
        patience: Union[int, None] = None,
    ):
        """Train the MLP head on cached DINOv2 features.

        Parameters
        ----------
        accelerator / precision / devices : forwarded to ``pl.Trainer``.
        extract_device / extract_precision : used for the one-shot feature
            extraction (defaults to CUDA + same precision as training when
            available, else CPU/fp32). Features are cached on disk and only
            computed once across train+predict.
        patience : int or None
            If ``None`` (default), no early stopping: train on *all* spots
            for ``epoch`` epochs and save the final-epoch weights. This is the
            right default for this pipeline because we predict on the same
            section we train on — there is no held-out generalization target,
            so restraining fit (via early stopping) only sacrifices training
            accuracy without buying anything. Pass a positive int to enable
            ``EarlyStopping(monitor='val_loss', patience=...)`` with a
            ``val_frac`` hold-out; the best-val-loss checkpoint is then loaded
            before persistence.
        val_frac : float
            Fraction of spots to hold out for validation when ``patience`` is
            set. Ignored when ``patience is None``.
        """
        pl.seed_everything(seed, workers=True)

        if extract_device is None:
            extract_device = "cuda" if torch.cuda.is_available() else "cpu"
        if extract_precision is None:
            extract_precision = "16-mixed" if extract_device == "cuda" else "32"

        feat = self._precompute_features(
            device=extract_device,
            precision=extract_precision,
            batch_size=feature_batch_size,
        )

        full_ds = CachedFeatureDataset(
            feat["spot_center"],
            feat["spot_neighbor"],
            feat["spot_labels"],
        )
        n_total = len(full_ds)
        if n_total < 2:
            raise RuntimeError(f"Too few spots ({n_total}) to train; need >=2")

        if patience is None:
            # Same-section inference → fit all spots as tightly as possible.
            train_loader = DataLoader(full_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            val_loader = None
            callbacks = []
            ckpt_cb = None
        else:
            if n_total < 4:
                raise RuntimeError(f"Too few spots ({n_total}) to split into train/val; need >=4")
            n_val = max(1, int(n_total * val_frac))
            n_train = n_total - n_val
            gen = torch.Generator().manual_seed(seed)
            train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=gen)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            ckpt_cb = ModelCheckpoint(
                monitor="val_loss",
                save_top_k=1,
                mode="min",
                filename="superres-{epoch:02d}-{val_loss:.3f}",
                dirpath=self._ckpt_dir,
            )
            callbacks = [
                EarlyStopping(monitor="val_loss", patience=patience, mode="min"),
                ckpt_cb,
            ]

        # logger=False keeps the cache dir tidy; LearningRateMonitor would need
        # a logger, so callers wanting LR curves pass their own.
        # enable_checkpointing mirrors ckpt_cb: False when there's no val split
        # (Lightning refuses ModelCheckpoint + enable_checkpointing=False), True
        # when our callback drives best-val-loss saving.
        trainer = pl.Trainer(
            max_epochs=epoch,
            precision=precision,
            accelerator=accelerator,
            devices=devices,
            callbacks=callbacks,
            gradient_clip_val=1.0,
            logger=False,
            enable_checkpointing=(ckpt_cb is not None),
        )
        trainer.fit(self.model, train_loader, val_loader)

        # With early stopping, load best-val-loss weights before persisting.
        # Without it, self.model already holds the final-epoch state.
        if ckpt_cb is not None:
            best = ckpt_cb.best_model_path
            if best and os.path.exists(best):
                best_state = torch.load(best, map_location="cpu", weights_only=False)["state_dict"]
                self.model.load_state_dict(best_state)
        trainer.save_checkpoint(os.path.join(self._ckpt_dir, "superres_model.ckpt"))
        self.train = False

    def run_superres(
        self, accelerator=None, precision="16-mixed", feature_batch_size: int = 64, predict_batch_size: int = 256
    ):
        """Predict per-subspot cell-type proportions on the high-res grid."""
        if accelerator is None:
            accelerator = "cuda" if torch.cuda.is_available() else "cpu"
        # Callers use Lightning's vocabulary ("gpu"); torch wants "cuda".
        extract_device = "cuda" if accelerator in ("gpu", "cuda") else accelerator
        extract_precision = "16-mixed" if extract_device == "cuda" else "32"

        # Features are cached on disk after the first run; if run_train already
        # populated self._features this is a free lookup.
        feat = self._precompute_features(
            device=extract_device,
            precision=extract_precision,
            batch_size=feature_batch_size,
        )
        sr_ds = CachedFeatureDataset(feat["sr_center"], feat["sr_neighbor"])
        sr_loader = DataLoader(sr_ds, batch_size=predict_batch_size, num_workers=0)

        device = torch.device(extract_device)
        pred = self._predict(sr_loader, device)
        logger.info(f"Assigning predictions to AnnData with shape {pred.shape}")
        self.sr_adata.obs[self.cell_type_name] = pred
        return self.sr_adata


class ImageTransform:
    def __init__(self, resize: int, mean: list[float], std: list[float]):
        self.resize = resize

        self.base_resize = transforms.Compose(
            [
                transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(resize),
            ]
        )

        self.to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

        self.augmentations = {
            "flip": transforms.Compose(
                [
                    transforms.RandomRotation(90),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                ]
            ),
            "noise": transforms.Compose(
                [
                    transforms.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 2.0)),
                    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
                ]
            ),
            "blur": transforms.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 2.0)),
            "dist": transforms.Compose(
                [
                    transforms.RandomAffine(degrees=30, shear=10),
                    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                ]
            ),
            "contrast": transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            "color": transforms.ColorJitter(hue=0.1),
            "crop": transforms.RandomResizedCrop(size=resize, scale=(0.5, 1.0)),
            "random": transforms.Compose(
                [
                    transforms.RandomApply(
                        [
                            transforms.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 2.0)),
                            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
                        ],
                        p=0.5,
                    ),
                    transforms.RandomApply(
                        [
                            transforms.RandomAffine(degrees=30, shear=10),
                            transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                        ],
                        p=0.5,
                    ),
                ]
            ),
        }

        self.valid_transform = transforms.Compose(
            [
                transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(
        self, img: Image.Image, phase: Literal["train", "valid"] = "train", param: str = "none"
    ) -> torch.Tensor:
        if phase == "train":
            img = self.base_resize(img)

            if param != "none":
                for para in param.split(","):
                    if para in self.augmentations:
                        img = self.augmentations[para](img)
                    else:
                        raise ValueError(f"Unknown augmentation parameter: {para}")

            img = self.to_tensor(img)

        elif phase == "valid":
            img = self.valid_transform(img)

        else:
            raise ValueError("phase must be 'train' or 'valid'")

        return img

    def transform_batch(self, imgs: Union[List[Image.Image], Image.Image], phase="train", param="none") -> torch.Tensor:
        if isinstance(imgs, Image.Image):
            imgs = [imgs]

        results = [self.__call__(img, phase=phase, param=param) for img in imgs]
        return torch.stack(results)  # shape: [B, C, H, W]


def cv2_detect_contour(
    img, CANNY_THRESH_1=100, CANNY_THRESH_2=200, apertureSize=5, L2gradient=True, all_cnt_info=False
):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 2:
        gray = (img * ((1, 255)[np.max(img) <= 1])).astype(np.uint8)
    else:
        logger.error("Image format error!")
    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2, apertureSize=apertureSize, L2gradient=L2gradient)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)
    cnt_info = []
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for c in cnts:
        cnt_info.append(
            (
                c,
                cv2.isContourConvex(c),
                cv2.contourArea(c),
            )
        )
    cnt_info = sorted(cnt_info, key=lambda c: c[2], reverse=True)
    if not cnt_info:
        raise RuntimeError("No tissue contour found in the image (Canny returned no closed edge).")
    cnt = cnt_info[0][0]
    if all_cnt_info:
        return cnt_info
    else:
        return cnt
