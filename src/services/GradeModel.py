from pathlib import Path
import os

from src.monitoring.logger import monitor_task_status
from utils.environ import set_huggingface_hf_env
set_huggingface_hf_env()
from sentence_transformers import SentenceTransformer, util


class DocumentGrader:
    """评估文档相关性"""
    def __init__(self,threshold:float=0.7):
        """
        :param threshold:  相关性阈值
        """
        local_model_path = os.getenv('HOME')+os.getenv('HF_MODELS_PATH')+"/models--BAAI--bge-large-zh-v1.5/snapshots/"
        # 获取最新的snapshot
        snapshots_dir = Path(local_model_path)
        if snapshots_dir.exists():
            # 通常只有一个snapshot目录，获取第一个
            snapshot_dirs = list(snapshots_dir.glob("*"))
            if snapshot_dirs:
                latest_snapshot = snapshot_dirs[0]
                model_path = str(latest_snapshot)
                # print(f"使用本地模型路径: {model_path}")
                self.model = SentenceTransformer(model_path)
            else:
                raise FileNotFoundError("未找到模型快照目录")
        else:
            self.model = SentenceTransformer('BAAI/bge-large-zh-v1.5')

        self.threshold = threshold

    def grade(self, question: str, docs: list) -> bool:
        if not docs:
            return False
        # 将文档内容合并
        combined_docs = "\n".join(docs)
        # 计算嵌入向量
        question_embedding = self.model.encode(question,convert_to_tensor=True)
        docs_embedding = self.model.encode(combined_docs,convert_to_tensor=True)

        # 计算余弦相似度
        similarity = util.pytorch_cos_sim(question_embedding,docs_embedding).item()
        monitor_task_status('doc score with question',similarity)
        return similarity >= self.threshold

    def get_similarity(self,question: str, answer: str) -> float:
        question_embedding = self.model.encode(question,convert_to_tensor=True)
        docs_embedding = self.model.encode(answer,convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(question_embedding,docs_embedding).item()
        return similarity


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    query = "黄独分布在哪些地区？"
    retrieved_docs = [
        '黄独（学名：）为薯蓣科薯蓣属的植物。多年生缠绕藤本。地下有球形或圆锥形块茎。叶腋内常生球形或卵圆形珠芽，大小不一，外皮黄褐色。心状卵形的叶子互生，先端尖锐，具有方格状小横脉，全缘，叶脉明显，7-9条，基出；叶柄基部扭曲而稍宽，与叶片等长或稍短。夏秋开花，单性，雌雄异株，穗状花序丛生。果期9-10月。分布于大洋洲、朝鲜、非洲、印度、日本、台湾、缅甸以及中国的江苏、广东、广西、安徽、江西、四川、甘肃、云南、湖南、西藏、河南、福建、浙江、贵州、湖北、陕西等地，生长于海拔300米至2,000米的地区，多生于河谷边、山谷阴沟或杂木林边缘，目前尚未由人工引种栽培。在美洲也可发现其踪迹，对美洲而言是外来种，有机会在农田大量繁殖，攀上高树争取日照。英文别名为air potato。黄药（本草原始），山慈姑（植物名实图考），零余子薯蓣（俄、拉、汉种子植物名称），零余薯（广州植物志、海南植物志），黄药子（江苏、安徽、浙江、云南等省药材名），山慈姑（云南楚雄）',
        '草黄树鸭（学名：\"\"）是分布在热带地区的树鸭属，包括中美洲、南美洲、撒哈拉以南非洲、印度次大陆及美国墨西哥湾沿岸地区。草黄树鸭长48-53厘米。牠们的喙是灰色的，头部及脚都很长，头部及上身呈浅黄色，两侧有些红色，冠深色，背部及双翼呈深灰色。尾巴及双翼上有栗褐色的斑纹，上尾有一白色的半月形，在飞时特别明显。雏鸟两侧对比较低，尾巴颜色也有不同。草黄树鸭很普遍。除了一些地区性的移动外，很多时都是留鸟，而在南欧也会有一些流浪者。牠们会在树枝上筑巢，有时也会在空心树中筑巢或利用其他鸟类的巢。牠们每次会生8-12只蛋。草黄树鸭喜欢栖息在淡水湖、稻田或水塘。牠们多在夜间觅食，主要吃种子及植物的其他部份。草黄树鸭是群居的，可以组成一大群。草黄树鸭是《非洲-欧亚大陆迁徙水鸟保护协定》所保护的物种之一。',
        '黄喉拟水龟（学名：、）又称柴棺龟、石金钱龟，为泽龟科拟水龟属的爬行动物。分布于中国的安徽、云南、海南、河南、广东、贵州、江西、浙江、福建、广西、香港及越南北部，台湾，日本波照间岛、西表岛、石垣岛及与那国岛，生活于低海拔地区的河流、湖泊等。头部为绿色，有黄色带状纹，上颌橄榄色，下颌黄色；背甲长12～14公分；有三条脊棱，中央一条发达，两侧常不明显，后缘呈锯齿状，灰褐色背面，黄色腹面，散步四角形的黑斑，四肢为橄榄色，具有绿色纵走带状纹；趾间具蹼。材棺龟体格强健，杂食性但偏好肉食，水陆两栖，但较喜欢在水边活动，食物包括叶菜、蟋蟀、面包虫、,鱼虾等。生性胆怯，但略具攻击性，会因为争食而攻击同类或其他龟类，有夜行倾向。雄性尾部粗大，泄殖孔超过背甲边缘，雌龟则尾部细短，泄殖孔距躯体较近。雌龟大于雄龟，每年可产卵1-3次，每次下1-5颗蛋，约65-80天孵化。冬天温度低于20度以下时活动会逐渐减少，最后进入冬眠状态。常见于丘陵地带半山区的山间盆地或河流谷地的水域中，常于附近的小灌丛或草丛中。性情害羞，夜行性乌龟。该物种的模式产地在浙江舟山群岛。在台湾普遍称为「柴棺龟」。',
        '黄尾魣（学名：），又称黄尾金梭鱼，俗名针梭、竹梭，为辐鳍鱼纲鲈形目鲭亚目金梭鱼科的其中一种。本鱼分布于印度西太平洋区，包括东非、红海、模里西斯、阿曼、印度、泰国、圣诞岛、可可群岛、马来西亚、日本、台湾、越南、印尼、菲律宾、巴布亚纽几内亚、香港、澳洲、马里亚纳群岛、新喀里多尼亚、关岛、萨摩亚群岛、东加等海域。该物种的模式产地在红海。水深3至50公尺。本鱼体延长呈鱼雷状，横切面几近圆柱形，侧线直而不弯曲；尾鳍深分叉，胸鳍末端达第一背鳍起点，腹鳍起点远前于第一背鳍起点。鳃盖具一尖棘，上颔骨远前于眼前缘。背部暗绿色，腹部银白色，尾鳍镶黑边，在侧线下具一暗色纵带，背鳍硬棘6枚；背鳍软条9枚；臀鳍硬棘2枚；臀鳍软条9枚，体长可达60公分。本鱼常成群地在礁湖或向海礁区的上方游走，属肉食性，以鱼类为食。为食用鱼，适合各种烹调方式。'
    ]
    print(DocumentGrader().grade(query,retrieved_docs))