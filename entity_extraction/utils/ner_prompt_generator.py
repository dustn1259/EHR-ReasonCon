import ast
from utils.utils import *
from utils.bm25_search_function import *

def list_to_string_with_custom_separator(list_obj):
    if isinstance(list_obj, list):
        return ";;".join(list_obj)
    return str(list_obj)


def generate_ner_prompt_json(line_num, content, numbered_note, ner_db, vertex_model,note):
    idx_container = []

    # Step 1: Sentence-based similarity search
    similar_records = find_similar_records(ner_db, 'sentence', content, n=5)
    for idx, score in similar_records:
        if score > 0:
            idx_container.append(idx)

    # Step 2: NER prediction using Gemini
    ner_prompt = open_file("./prompt/ner.txt").replace("<<<Input_sentencee>>>", content)

    response = vertex_model.models.generate_content(model="gemini-2.0-flash-001",contents=ner_prompt)

    try:
        ner_gemini_list = ast.literal_eval(response.text)
    except:
        ner_gemini_list = []  # fallback

    # Step 3: Entity-based similarity search
    for ner_ent in ner_gemini_list:
        similar_records2 = find_similar_records(ner_db, 'extracted_entities', ner_ent, n=5)
        for idx, score in similar_records2:
            if score > 0:
                idx_container.append(idx)

    ### seems to output the final idx_container
    # Step 4: Generate the few-shot prompt
    # Set to track processed sentences
    processed_sentences = set()

    # Improved format - numbered example format and clear target sentence distinction
    ner_shots = []
    example_count = 1  # Add example number counter
    
    ner_shot_format = '''
## Example {example_number}

**Target Sentence:**
```
Line <<<line_number>>>: <<<sentence>>>
```

**Context (Additional Clinical Note Lines):**
```
<<<content>>>
```

**Your Task:** Extract entities ONLY from the target sentence (Line <<<line_number>>>)

**Output:**
```
<<<entity_pairs>>>
```
'''

    for idx in idx_container:
        try:
            line_number = str(ner_db['line_number'][idx])
            sentence = ner_db['sentence'][idx]
            contents_ = ner_db['content'][idx]
       
            # Skip if the sentence has already been processed
            sentence_line_pair = (sentence, line_number,contents_)
            if sentence_line_pair in processed_sentences:
                continue
            
            # Mark as a processed sentence
            processed_sentences.add(sentence_line_pair)

            # Fetch and format the context (use the full note content)
            content_text = ner_db['content'][idx] if 'content' in ner_db and ner_db['content'][idx] else sentence
            
            # Find all entity pairs related to this sentence
            entity_pairs_text = ""
            # Iterate over the entire ner_db and find all entities with the same sentence
            for db_idx in range(len(ner_db['sentence'])):
                db_line_number = str(ner_db['line_number'][db_idx])
                db_content = ner_db['content'][db_idx]  
                if (ner_db['sentence'][db_idx] == sentence and
                    db_line_number == line_number and 
                    db_content == content_text): # If it is the same sentence
                    try:
                        # Use the new function instead of the existing list_to_string
                        extracted = list_to_string_with_custom_separator(ast.literal_eval(ner_db['extracted_entities'][db_idx]))
                        gold = list_to_string_with_custom_separator(ner_db['gold_entities'][db_idx])
                        
                        # Add only if the value is not empty
                        if extracted and gold:
                            entity_pairs_text += f"{{{extracted}: {gold}}}\n"
                    except Exception as inner_e:
                        print(f"Error processing entity at idx {db_idx}: {inner_e}")
            
            # Remove the last newline
            if entity_pairs_text:
                entity_pairs_text = entity_pairs_text.rstrip("\n")
            else:
                # If there are no entities, process in the existing way
                try:
                    extracted = list_to_string_with_custom_separator(ast.literal_eval(ner_db['extracted_entities'][idx]))
                    gold = list_to_string_with_custom_separator(ner_db['gold_entities'][idx])

                    if extracted and gold:
                        entity_pairs_text = f"{{{extracted}: {gold}}}"
                    else:
                        entity_pairs_text = "{:}"
                except:
                    entity_pairs_text = "{:}"
            
            # Add example number formatting
            shot = ner_shot_format.format(example_number=example_count).replace("<<<line_number>>>", line_number)\
                                .replace("<<<sentence>>>", sentence)\
                                .replace("<<<content>>>", content_text)\
                                .replace("<<<entity_pairs>>>", entity_pairs_text)
            
            ner_shots.append(shot)
            example_count += 1  # Increase the counter for the next example
            
        except Exception as e:
            print(f"Error creating shot for idx {idx}: {e}")
    # Step 5: Generate the final prompt

    
    final_prompt = open_file("./prompt/new_ner.txt")
    final_prompt += "\n".join(ner_shots)
    # Format the current query - clearer structure and emphasis on the target sentence
    final_prompt += f'''

## Your Task Now

**Target Sentence:**
```
Line {line_num}: {content}
```

**Context (Additional Clinical Note Lines):**
```
{note}
```

**Your Task:** Extract entities ONLY from the target sentence (Line {line_num})
Please use the provided clinical note for reference only.

**Expected Output Format:**
```
{{clinical term from the target sentence: associated value}}
```

Extracted entities:
'''
    return final_prompt