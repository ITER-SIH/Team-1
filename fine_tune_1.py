import json
import tempfile
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

with open("E:/JURIMIND/v0.0.1/ipc_reduced.json", 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

tokenizer.add_special_tokens({'pad_token': '[PAD]'})


filtered_data = [example for example in data if example.get("train") == True]


formatted_data = ""
for item in filtered_data:
    formatted_data += f"Chapter {item['chapter']}: {item['chapter_title']}\n"
    formatted_data += f"Section {item['Section']}: {item['section_title']}\n"
    formatted_data += f"Description: {item['section_desc']}\n\n"


inputs = tokenizer(formatted_data, return_tensors="pt", max_length=512, truncation=True, padding=True)


with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as temp_file:
    temp_file.write(formatted_data)

dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=temp_file.name,
    block_size=128
)


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./fine_tuned_model",  
    overwrite_output_dir=True,
    num_train_epochs=100, 
    per_device_train_batch_size=20,
    save_steps=10_000,
    save_total_limit=2,
    evaluation_strategy="steps",
    eval_steps=10_000,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

model.save_pretrained("./fine_tuned_model")

tokenizer.save_pretrained("./fine_tuned_model")
