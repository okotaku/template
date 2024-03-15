CONFIG=$1
filename=$(basename "$CONFIG")
filename_noextension="${filename%.*}"

for sub in backpack backpack_dog bear_plushie berry_bowl can candle cat cat2 clock colorful_sneaker dog dog2 dog3 dog5 dog6 dog7 dog8 duck_toy fancy_boot grey_sloth_plushie; do
    diffengine train $CONFIG \
        --cfg-options train_dataloader.dataset.subject=$sub \
        --work-dir work_dirs/$filename_noextension/$sub
done
diffengine analyze mean_score work_dirs/$filename_noextension/
