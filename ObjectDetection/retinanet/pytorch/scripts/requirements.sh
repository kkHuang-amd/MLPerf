echo "-------------------------------------------------------"
echo "Installing dependancies"
echo "-------------------------------------------------------"
python3 -m pip install -r requirements.txt --user

echo "-------------------------------------------------------"
echo "Pull backbone"
echo "-------------------------------------------------------"
bash ./scripts/download_backbone.sh

