 cd ../
 zip -r AnomalyVPT0419.zip ./AnomalyVPT/ \
 -x="AnomalyVPT/output/*" \
 -x="AnomalyVPT/output_npu/*" \
 -x="AnomalyVPT/pretrained/*" \
 -x="AnomalyVPT/kernel_*" \
 -x="AnomalyVPT/.idea/*" \
 -x="AnomalyVPT/.ipynb_checkpoints/*" \
 -x="AnomalyVPT/*.ipynb" \
 -x="AnomalyVPT/core" \
 -x="AnomalyVPT/pretrained/*"  \
 -x="*.jpg" \
 -x="*.pth.tar"