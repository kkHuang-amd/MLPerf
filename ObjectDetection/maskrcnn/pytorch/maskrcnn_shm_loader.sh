mkdir -p /dev/shm/data
mkdir -p /dev/shm/data/coco2017
#cp -r ~/data/coco2017/train2017 /dev/shm/data/coco2017/
#cp -r ~/data/coco2017/val2017 /dev/shm/data/coco2017/
#cp -r ~/data/coco2017/annotations /dev/shm/data/coco2017/
#cp -r ~/data/coco2017/models/ /dev/shm/data/coco2017/

cp -r $1/train2017 /dev/shm/data/coco2017/
cp -r $1/val2017 /dev/shm/data/coco2017/
cp -r $1/annotations /dev/shm/data/coco2017/
cp -r $1/models/ /dev/shm/data/coco2017/
