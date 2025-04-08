from illufly_tts.g2p.chinese_g2p import ChineseG2P
from misaki import zh

def compare_g2p_outputs(text):
    """比较两个G2P模型的输出差异"""
    your_g2p = ChineseG2P()
    misaki_g2p = zh.ZHG2P(version="1.1")
    
    your_result = your_g2p.text_to_phonemes(text)
    misaki_result, _ = misaki_g2p(text)
    
    print(f"\n{'=' * 50}")
    print(f"输入文本: {text}")
    print(f"{'=' * 50}")
    print(f"你的结果: {your_result}")
    print(f"Misaki结果: {misaki_result}")
    
    # 计算差异
    if your_result != misaki_result:
        print(f"\n[!] 检测到差异")
    else:
        print(f"\n[√] 结果一致")

def main():
    test_cases = [
        "你好世界",
        "今天天气真不错",
        "一个苹果两块钱",
        "他不是中国人",
        "银行行长来了",
        "我今天要去买一些东西，不知道商店几点开门",
        "这是一个测试，包含标点符号！",
        "电影《长津湖》非常感人",
        "小明今天看了一本书，书名叫《三国演义》",
        "12345上山打老虎"
    ]
    
    print("开始比较ChineseG2P与Misaki的zh.ZHG2P输出差异...\n")
    
    for test_text in test_cases:
        compare_g2p_outputs(test_text)
    
    print("\n比较完成")

if __name__ == "__main__":
    main()