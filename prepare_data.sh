
input_root=../data/raw
output_root=../data/cropped
dataset=uw3
split=test

data_path=${input_root}/${dataset}/${split}
out_data_path=${output_root}/${dataset}/${split}

python prepare_data.py -img ${data_path}/images/ -xml ${data_path}/gt/ -ocr ${data_path}/ocr/ -o ${out_data_path}
