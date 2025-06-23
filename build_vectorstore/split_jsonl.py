import os

def split_jsonl(input_path, output_dir, lines_per_file=200_000):
    os.makedirs(output_dir, exist_ok=True)
    with open(input_path, 'r', encoding='utf-8') as infile:
        count = 0
        file_index = 0 # Start with part_0
        out_file = open(os.path.join(output_dir, f'part_{file_index}.jsonl'), 'w', encoding='utf-8')
        for line in infile:
            if count > 0 and count % lines_per_file == 0:
                out_file.close()
                file_index += 1
                out_file = open(os.path.join(output_dir, f'part_{file_index}.jsonl'), 'w', encoding='utf-8')
            out_file.write(line)
            count += 1
        out_file.close()

split_jsonl('../wiki18_400w.jsonl', './splits/')
