# dino_timm.py
#
# Кодировщик картинки DINOv3 ViT-L из timm вместо закрытого входа facebook/.
#
# TRELLIS.2 берёт DINOv3 из facebook/dinov3-vitl16-pretrain-lvd1689m, а Meta
# открывает этот репозиторий только после ручной проверки заявки. Лицензия
# DINOv3 разрешает принявшим её распространять модель дальше, и Hugging Face
# выложил те же веса в timm (timm/vit_large_patch16_dinov3.lvd1689m) без
# ворот, со ссылкой на лицензию Meta. Условия лицензии на нас действуют так
# же («Built with DINOv3» на странице «О сервисе»).
#
# Признаки считаются ровно как в TRELLIS.2 (DinoV3FeatureExtractor): все
# блоки, затем layer_norm без весов по последнему скрытому состоянию.
# Проверено 21.10.2026 на 512 и 1024 точках против копии в формате
# transformers: косинус по токенам >= 0.99998, средняя разница ~3e-4.
#
# Регистрируется в trellis2.modules.image_feature_extractor под именем
# TimmDinoV3FeatureExtractor; pipeline.json указывает на него.

from typing import List, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

TIMM_NAME = "vit_large_patch16_dinov3.lvd1689m"


class TimmDinoV3FeatureExtractor:
    def __init__(self, model_name: str = TIMM_NAME, image_size: int = 512):
        import timm

        self.model_name = model_name
        # dynamic_img_size: TRELLIS.2 зовёт кодировщик и на 512, и на 1024
        self.model = timm.create_model(model_name, pretrained=True, dynamic_img_size=True)
        # timm держит периоды RoPE во float32, оригинал — в bfloat16;
        # приводим, чтобы признаки совпали с оригиналом (карточка модели timm)
        for mod in self.model.modules():
            if isinstance(getattr(mod, "periods", None), torch.Tensor):
                mod.periods = mod.periods.to(torch.bfloat16).to(torch.float32)
        self.model.eval()
        self.image_size = image_size
        self.transform = transforms.Compose(
            [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

    def to(self, device):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()

    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        m = self.model
        image = image.to(m.patch_embed.proj.weight.dtype)
        x = m.patch_embed(image)
        x, rope = m._pos_embed(x)
        x = m.norm_pre(x)
        for blk in m.blocks:
            x = blk(x, rope=rope)
        return F.layer_norm(x, x.shape[-1:])

    @torch.no_grad()
    def __call__(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((self.image_size, self.image_size), Image.LANCZOS) for i in image]
            image = [np.array(i.convert("RGB")).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).cuda()
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        image = self.transform(image).cuda()
        return self.extract_features(image)


def register():
    """Сделать класс видимым для pipeline.json TRELLIS.2."""
    import trellis2.modules.image_feature_extractor as fe

    fe.TimmDinoV3FeatureExtractor = TimmDinoV3FeatureExtractor
