# scAFGCC
！[scAFGCC][]
## Testing Commands
We have constructed four graphs for example, including Human3, Darmanis, Klein, and Zeisel.

### For Human3
```
python main.py --dataset h5ad/real_data/baron3_preprocessed.h5ad --embedder scAFGCC-Results --layers [512] --pred_hid 1024 --lr 0.001 --topk 3 --device 0
```
### For Darmanis
```
python main.py --dataset h5ad/real_data/darmanis_preprocessed.h5ad --embedder scAFGCC-Results --layers [512] --pred_hid 1024 --lr 0.001 --topk 3 --device 0
```
### For Klein
```
python main.py --dataset h5ad/real_data/klein_preprocessed.h5ad --embedder scAFGCC-Results --layers [512] --pred_hid 1024 --lr 0.001 --topk 3 --device 0
```
### For Zeisel
```
python main.py --dataset h5ad/real_data/zeisel_preprocessed.h5ad --embedder scAFGCC-Results --layers [512] --pred_hid 1024 --lr 0.001 --topk 3 --device 0
```
