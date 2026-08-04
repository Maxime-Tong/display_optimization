from collections import defaultdict
import json

path = "D:\\workspace\\master\\3DGS\\Vulkan\\screen_adaptor_v2\\outputs\\scene_manifest.assignments.json"
with open(path, "r") as f:
    data = json.load(f)

clusters = defaultdict(list)
for data_path, cluster_id in data.items():
    clusters[cluster_id].append(data_path)
print(clusters[0])