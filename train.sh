suffix=/icdar
augment=false

root_data_path=../data/cropped
data_path=${root_data_path}/${suffix}/train
out_path=$suffix

if [ "$augment" = true ]; then
  out_path=${out_path}-augmented
  args='--augment_tables'
fi

out_path=./trainings/${out_path}_$(date +'%d-%m-%Y')

# rm -r ${out_path}
python train.py -img ${data_path}/images/ -gt ${data_path}/gt/ -ocr ${data_path}/ocr/ -o ${out_path} -e 200 ${args} --dr 0.8
