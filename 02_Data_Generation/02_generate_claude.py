import pandas as pd
import os
from dotenv import load_dotenv
from anthropic import Anthropic
import time

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Load PERSUADE with topics
print("Loading PERSUADE with topics...")
persuade_df = pd.read_csv('data/raw/persuade_with_topics.csv')

generated_essays = []

print(f"\n🚀 Generating 1,000 Claude essays with MATCHED topics...")
print("This will take approximately 15-20 minutes.\n")

for idx in range(1000):
    human_essay = persuade_df.iloc[idx]['full_text']
    essay_length = len(human_essay)
    topic = persuade_df.iloc[idx]['extracted_topic']
    
    prompt = f"Write an argumentative essay about {topic} that is approximately {essay_length} characters long."
    
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        generated_text = response.content[0].text
        
        generated_essays.append({
            'essay_id': idx,
            'source_id': persuade_df.iloc[idx]['text_id'],
            'topic': topic,
            'target_length': essay_length,
            'generated_text': generated_text,
            'actual_length': len(generated_text)
        })
        
        if (idx + 1) % 50 == 0:
            print(f"✅ Generated {idx + 1}/1000 essays...")
        
        time.sleep(0.5)
        
    except Exception as e:
        print(f"❌ Error at essay {idx}: {e}")
        generated_essays.append({
            'essay_id': idx,
            'source_id': persuade_df.iloc[idx]['text_id'],
            'topic': topic,
            'target_length': essay_length,
            'generated_text': f"GENERATION_FAILED: {str(e)}",
            'actual_length': 0
        })

result_df = pd.DataFrame(generated_essays)
result_df.to_csv('data/raw/claude_samples.csv', index=False)

print(f"\n✅ COMPLETE! Generated {len(result_df)} essays")
print(f"Saved to: data/raw/claude_samples.csv")
successful = len(result_df[~result_df['generated_text'].str.contains('FAILED', na=False)])
print(f"Success rate: {successful/len(result_df)*100:.1f}%")