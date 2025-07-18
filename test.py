# test_api.py
import os
from openai import OpenAI

# 设置API配置
client = OpenAI(
    api_key="####",
    base_url="https://idealab.alibaba-inc.com/api/openai/v1"
)

try:
    # 测试API调用
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello, test"}],
        max_tokens=10
    )
    print("✅ API测试成功:")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"❌ API测试失败: {e}")
    print("尝试其他模型名称...")
    
    # 尝试其他可能的模型名称
    models_to_try = ["gpt-4", "gpt-3.5-turbo-16k", "text-davinci-003"]
    for model in models_to_try:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            print(f"✅ 模型 {model} 可用")
            break
        except Exception as e2:
            print(f"❌ 模型 {model} 不可用: {e2}")
