import * as tf from '@tensorflow/tfjs';
import { Rank, Tensor } from '@tensorflow/tfjs';
import { bundleResourceIO } from '@tensorflow/tfjs-react-native';

export class FaceBoxes {
  enlarge: number;
  model: tf.GraphModel;

  constructor(model: tf.GraphModel) {
    this.enlarge = 1.1;
    this.model = model;
  }

  async estimateFaces(input: Tensor<Rank.R3>, size=128) {
    const results: Array<{ confidence: number, box: any, boxRaw: any, image: any }> = [];
    const resizeT = tf.image.resizeBilinear(input, [size, size]);
    // const castT = resizeT.toInt();

    const [scoresT, boxesT, numT] = (this.model.execute(resizeT.reshape([1, size, size, 3])) as tf.Tensor<tf.Rank>[]);
    const scores = scoresT.dataSync();
    const squeezeT = boxesT.squeeze();
    const boxes = squeezeT.arraySync() as number[][];
    scoresT.dispose();
    boxesT.dispose();
    squeezeT.dispose();
    numT.dispose();
    // castT.dispose();
    resizeT.dispose();
    for (const i in boxes) {
      const crop = [boxes[i][0] / this.enlarge, boxes[i][1] / this.enlarge, boxes[i][2] * this.enlarge, boxes[i][3] * this.enlarge];
      const boxRaw = [crop[1], crop[0], (crop[3]) - (crop[1]), (crop[2]) - (crop[0])];
      const box = [
        parseInt((boxRaw[0] * input.shape[2]).toString()),
        parseInt((boxRaw[1] * input.shape[1]).toString()),
        parseInt((boxRaw[2] * input.shape[2]).toString()),
        parseInt((boxRaw[3] * input.shape[1]).toString())];
      console.log('before resize')
      const resized = tf.image.cropAndResize((input as any).reshape([1, 64, 64, 3]), [crop], [0], [size, size]);
      console.log('after resize')
      const image = resized.div([size - 1]);
      resized.dispose();
      results.push({ confidence: scores[i], box, boxRaw, image });
      // add mesh, meshRaw, annotations,
    }
    return results;
  }
}

export async function load() {
  const modelJson =  require('./assets/models/blazeface-front.json');
  const modelWeights = require('./assets/models/blazeface-front.bin');
  const model = await tf.loadGraphModel(bundleResourceIO(modelJson, modelWeights));
  const faceboxes = new FaceBoxes(model);
  return faceboxes;
}
