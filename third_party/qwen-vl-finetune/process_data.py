import jsonlines
from pathlib import Path

import random
random.seed(998244353)
rootdir = None

CASE_NUM = 2

def get_L1_data():
    global rootdir
    rootdir = Path("data")
    with jsonlines.open("data/dataset_l1_training-ref_seed_prompts-refined-ref_seed_prompts-img.jsonl", "r") as reader:
        df = list(reader)
        
    for row in df:
        row["ref_seed_prompts_img_path"] = [p.replace("images/", "images-L1/") for p in row["ref_seed_prompts_img_path"]]
    return df

def get_L2_data():
    global rootdir
    rootdir = Path("benchmark/L2-benchmark")
    with jsonlines.open(rootdir / "captions.jsonl", "r") as reader:
        caption_list = list(reader)
        caption_dict = {item["image_path"]: item["caption"] for item in caption_list}
    
    with jsonlines.open(rootdir / "train_agent_meta.jsonl", "r") as reader:
        metadata = list(reader)
        
    df = []
    for agent in metadata:
        row = {
            "ref_seed_prompts_img_path": [],
            "ref_seed_prompts": [],
        }
        for img in agent["images"]:
            img_path = Path(f"benchmark/{img['image_path']}").relative_to(rootdir)
            img_key = f"L2-benchmark/{img_path}"
            if img_key not in caption_dict:
                print(img_key)
                continue
            row["ref_seed_prompts_img_path"].append(str(img_path))
            row["ref_seed_prompts"].append(caption_dict[img_key])
            
        df.append(row)
    return df

import os
import shutil

def main():
    import random
    random.seed(998244353)
    df = get_L1_data()
    
    results = []
    skip_cnt = 0
    
    save_img_dir = Path("data/images")
    
    os.makedirs(save_img_dir, exist_ok=True)
    
    agent_id = 0
    
    for row in df:
        img_pool = row["ref_seed_prompts_img_path"]
        img_pool = list(filter(lambda x: (rootdir / x).exists(), img_pool))
        if len(img_pool) < 3:
            skip_cnt += 1
            print(skip_cnt, len(img_pool))
            continue
        
        for img in img_pool:
            assert (rootdir / img).exists()
        
        for i in range(len(img_pool)):
            img = img_pool[i]
            src = rootdir / img
            dst = save_img_dir / f"{agent_id}" / Path(img).name
            os.makedirs(dst.parent, exist_ok=True)
            shutil.copy(src, dst)
            img_pool[i] = str(dst)
        
        agent_id += 1
        
        for i in range(len(img_pool)):
            rem_img_pool = img_pool[:i] + img_pool[i+1:]
            for j in range(CASE_NUM):
                results.append({
                    "ref_images": random.sample(rem_img_pool, random.randint(2, min(5, len(rem_img_pool)))),
                    "prompt": row["ref_seed_prompts"][i],
                    "gt_images": img_pool[i],
                })
                
    df = get_L2_data()
    
    for row in df:
        img_pool = row["ref_seed_prompts_img_path"]
        img_pool = list(filter(lambda x: (rootdir / x).exists(), img_pool))
        if len(img_pool) < 3:
            skip_cnt += 1
            print(skip_cnt, len(img_pool))
            continue
        
        for img in img_pool:
            assert (rootdir / img).exists()
            
        for i in range(len(img_pool)):
            img = img_pool[i]
            src = rootdir / img
            dst = save_img_dir / f"{agent_id}" / Path(img).name
            os.makedirs(dst.parent, exist_ok=True)
            shutil.copy(src, dst)
            img_pool[i] = str(dst)
        
        agent_id += 1
        
        for i in range(len(img_pool)):
            rem_img_pool = img_pool[:i] + img_pool[i+1:]
            for j in range(CASE_NUM):
                results.append({
                    "ref_images": random.sample(rem_img_pool, random.randint(2, min(5, len(rem_img_pool)))),
                    "prompt": row["ref_seed_prompts"][i],
                    "gt_images": img_pool[i],
                })
            
            
    random.shuffle(results)
    print(f"Total cases: {len(results)}, {agent_id} agents.")
    with jsonlines.open("data/metadata-full.jsonl", "w") as writer:
        writer.write_all(results)
            
        
if __name__ == "__main__":
    main()