# Quick train: single pass, no clustering
# For the full pipeline with clustering, use:
#   python -m src.screen_adaptor.pipeline full --data-dir ... --output-dir outputs --clusters 8 --device cuda
python -m src.screen_adaptor.pipeline pretrain --data-dir D:\workspace\master\3DGS\Vulkan\screen_adaptor_v2\datasets --device cuda --output-dir output
python -m src.screen_adaptor.pipeline train --data-dir D:\workspace\master\3DGS\Vulkan\screen_adaptor_v2\datasets --device cuda --output-dir output