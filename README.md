# RegO

This is the official project of the Region-Based Optimization (RegO) method proposed by our paper titled "Region-Based Optimization in Continual Learning for Audio Deepfake Detection", published on the the 39th Annual AAAI Conference on Artificial Intelligence (AAAI 2025). To facilitate open source sharing, we have integrated the main method implementation into the CLEAR Benchmark !!!!!!!!!!!

### INTRODUCTION
The rapid advancements in speech synthesis and voice conversion technologies have brought convenience but also significant security risks, emphasizing the urgent need for effective audio deepfake detection. While existing models perform well in controlled settings, their effectiveness diminishes when faced with the diverse and evolving nature of real-world deepfakes. We propose Region-Based Optimization (RegO), a novel continual learning method for audio deepfake detection. RegO utilizes the Fisher Information Matrix (FIM) to identify critical neuron regions, dividing them into four categories. It applies a region-adaptive strategy: fine-tuning less important regions for quick adaptation, parallel optimization for real audio regions, orthogonal optimization for fake audio regions, and adaptive optimization for shared regions to balance memory stability and learning plasticity. Additionally, the Ebbinghaus Forgetting Mechanism releases redundant neurons from previous tasks, enhancing adaptability and enabling the model to learn more generalized discriminative features.

### RUN！！！！！！！ 🏃
```
python train.sh --yaml

### Acknowledgements
```bibtex
[CLEAR!!!!!!](https://clear-benchmark.github.io/)
@inproceedings{lin2021clear,
  title={The clear benchmark: Continual learning on real-world imagery},
  author={Lin, Zhiqiu and Shi, Jia and Pathak, Deepak and Ramanan, Deva},
  booktitle={Thirty-fifth conference on neural information processing systems datasets and benchmarks track (round 2)},
  year={2021}
}
```
