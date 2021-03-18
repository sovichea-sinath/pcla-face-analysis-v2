
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import React from 'react';
import {ActivityIndicator, Button, StyleSheet, View, Platform } from 'react-native';
import Svg, { Circle, Rect, G, Line, Text} from 'react-native-svg';

import * as Permissions from 'expo-permissions';
import { Camera } from 'expo-camera';
import { ExpoWebGLRenderingContext } from 'expo-gl';

import * as tf from '@tensorflow/tfjs';
import * as blazeface from '@tensorflow-models/blazeface';
import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';
import {cameraWithTensors, bundleResourceIO} from '@tensorflow/tfjs-react-native';
import { Rank, Tensor } from '@tensorflow/tfjs';
import * as HumanFace from './face'

interface ScreenProps {
  returnToMain: () => void;
}

interface ScreenState {
  hasCameraPermission?: boolean;
  // tslint:disable-next-line: no-any
  cameraType: any;
  isLoading: boolean;
  // tslint:disable-next-line: no-any
  faceDetector?: blazeface.BlazeFaceModel;
  emotionDetector?: tf.LayersModel;
  faces?: blazeface.NormalizedFace[];
  predictions?: any[];
}

const inputTensorWidth = 152;
const inputTensorHeight = 200;

const AUTORENDER = true;

// tslint:disable-next-line: variable-name
const TensorCamera = cameraWithTensors(Camera);

// emotion types.
const emotionTypes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'];

export default class RealtimeDemo extends React.Component<ScreenProps,ScreenState> {
  rafID?: number;

  constructor(props: ScreenProps) {
    super(props);
    this.state = {
      isLoading: true,
      cameraType: Camera.Constants.Type.front,
    };
    this.handleImageTensorReady = this.handleImageTensorReady.bind(this);
  }

  async loadBlazefaceModel() {
    const model =  await blazeface.load();
    return model;
  }

  // load emotion model.
  async loadEmotionModel() {
    const modelJson =  require('./assets/models/emotion/model.json');
    const modelWeights = require('./assets/models/emotion/data.bin');
    const model = await tf.loadLayersModel(bundleResourceIO(modelJson, modelWeights));
    return model;
  }

  async handleImageTensorReady(
    images: IterableIterator<tf.Tensor3D>,
    updatePreview: () => void, gl: ExpoWebGLRenderingContext) {
    const loop = async () => {
      if(!AUTORENDER) {
        updatePreview();
      }

      if (this.state.faceDetector != null) {
        const imageTensor = images.next().value;
        const returnTensors = false;
        const faces = await this.state.faceDetector.estimateFaces(
          imageTensor, returnTensors);
        this.setState({faces});
        // get the tensor from blaze face.
        let facesTensor: any = await this.state.faceDetector.estimateFaces(
          imageTensor, !returnTensors);
        // crop and grayscale the faces.
        facesTensor = facesTensor.map((face: any) => {
          const topLeft = face.topLeft as tf.Tensor1D;
          const bottomRight = face.bottomRight as tf.Tensor1D;
          // const width = Math.floor((bottomRight.dataSync()[0] - topLeft.dataSync()[0]));
          // const height = Math.floor((bottomRight.dataSync()[1] - topLeft.dataSync()[1]));
          const boxes: Tensor<Rank.R2> = tf.concat([topLeft, bottomRight]).reshape([-1, 4]);
          const faceTensor = tf.image.cropAndResize(imageTensor.reshape([1, inputTensorHeight, inputTensorWidth, 3]), boxes, [0], [48, 48]);
          // grayscale the image.
          // the vector to normalize the channel in the order [red, green, blue].
          const grayMatrix = [0.2989, 0.5870, 0.1140];
          // devide the face into 3 differnce color channel.
          const [red, green, blue] = tf.split(faceTensor, 3, 3);
          faceTensor.dispose();
          // normalize the channels with the gray matrix.
          const redNorm = tf.mul(red, grayMatrix[0]);
          const greenNorm = tf.mul(green, grayMatrix[1]);
          const blueNorm = tf.mul(blue, grayMatrix[2]);
          // dispose the normal channels.
          red.dispose();
          green.dispose();
          blue.dispose();
          // grayscale the image
          const grayscale = tf.addN([redNorm, greenNorm, blueNorm]);
          redNorm.dispose();
          greenNorm.dispose();
          blueNorm.dispose();
          const normalize = tf.tidy(() => grayscale.sub(0.5).mul(2));
          grayscale.dispose();
          return normalize;
        })
        tf.dispose(imageTensor);

        // get the prediction.
        const predictions = facesTensor.map((faceTensor: any) => {
          const predictObj: any = {}
          if (this.state.emotionDetector) {
            // call the emotion prediction and get it data.
            const emotionProb = (this.state.emotionDetector.predict(faceTensor) as any)
              .dataSync()
            ;
            // return the percentage and and it type.
            const maxProp = Math.max(...emotionProb);
            // console.log('emotionProb', emotionProb);
            predictObj.emotions = {
              emotion: emotionTypes[emotionProb.indexOf(maxProp)],
              percentage: maxProp
            }
          }
          faceTensor.dispose();
          return predictObj;
        });
        tf.dispose(facesTensor)
        this.setState({predictions})
        console.log('predictions', this.state.predictions);
      }

      if(!AUTORENDER) {
        gl.endFrameEXP();
      }
      this.rafID = requestAnimationFrame(loop);
    };

    loop();
  }

  componentWillUnmount() {
    if(this.rafID) {
      cancelAnimationFrame(this.rafID);
    }
  }

  async componentDidMount() {
    await tf.ready();
    const { status } = await Permissions.askAsync(Permissions.CAMERA);

    const [
      blazefaceModel,
      emotionModel
    ] = await Promise.all([
      this.loadBlazefaceModel(),
      this.loadEmotionModel()
    ]);

    this.setState({
      hasCameraPermission: status === 'granted',
      isLoading: false,
      faceDetector: blazefaceModel,
      emotionDetector: emotionModel,
    });
  }

  renderFaces() {
    const {faces} = this.state;
    if(faces != null) {
      const faceBoxes = faces.map((f, fIndex) => {
        const topLeft = f.topLeft as number[];
        const bottomRight = f.bottomRight as number[];

        const landmarks = (f.landmarks as number[][]).map((l, lIndex) => {
          return <Circle
            key={`landmark_${fIndex}_${lIndex}`}
            cx={l[0]}
            cy={l[1]}
            r='2'
            strokeWidth='0'
            fill='blue'
            />;
        });

        return <G key={`facebox_${fIndex}`}>
          <Rect
            x={topLeft[0]}
            y={topLeft[1]}
            strokeWidth={5}
            fill={'red'}
            fillOpacity={0.1}
            width={(bottomRight[0] - topLeft[0])}
            height={(bottomRight[1] - topLeft[1])}
          />
          {landmarks}
          {/* <Text
            fill="none"
            stroke="purple"
            fontSize="20"
            fontWeight="bold"
            x={topLeft[0]}
            y={topLeft[1]}
            textAnchor="middle"
            scale={1}
          > text
          </Text> */}
        </G>;
      });

      const flipHorizontal = Platform.OS === 'ios' ? 1 : -1;
      return <Svg height='100%' width='100%'
        viewBox={`0 0 ${inputTensorWidth} ${inputTensorHeight}`}
        scaleX={flipHorizontal}>
          {faceBoxes}
        </Svg>;
    } else {
      return null;
    }
  }

  render() {
    const {isLoading} = this.state;

    // TODO File issue to be able get this from expo.
    // Caller will still need to account for orientation/phone rotation changes
    let textureDims: { width: number; height: number; };
    if (Platform.OS === 'ios') {
        textureDims = {
          height: 1920,
          width: 1080,
        };
      } else {
        textureDims = {
          height: 1200,
          width: 1600,
        };
      }

    const camView = <View style={styles.cameraContainer}>
      <TensorCamera
        // Standard Camera props
        style={styles.camera}
        type={this.state.cameraType}
        zoom={0}
        // tensor related props
        cameraTextureHeight={textureDims.height}
        cameraTextureWidth={textureDims.width}
        resizeHeight={inputTensorHeight}
        resizeWidth={inputTensorWidth}
        // resizeHeight={256}
        // resizeWidth={256}
        resizeDepth={3}
        onReady={this.handleImageTensorReady}
        autorender={AUTORENDER}
      />
      <View style={styles.modelResults}>
        {this.renderFaces()}
      </View>
    </View>;

    return (
      <View style={{width:'100%'}}>
        <View style={styles.sectionContainer}>
          <Button
            onPress={this.props.returnToMain}
            title='Back'
          />
        </View>
        {isLoading ? <View style={[styles.loadingIndicator]}>
          <ActivityIndicator size='large' color='#FF0266' />
        </View> : camView}
      </View>
    );
  }

}

const styles = StyleSheet.create({
  loadingIndicator: {
    position: 'absolute',
    top: 20,
    right: 20,
    zIndex: 200,
  },
  sectionContainer: {
    marginTop: 32,
    paddingHorizontal: 24,
  },
  cameraContainer: {
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'center',
    alignItems: 'center',
    width: '100%',
    height: '100%',
    backgroundColor: '#fff',
  },
  camera : {
    position:'absolute',
    left: 50,
    top: 100,
    width: 600/2,
    height: 800/2,
    zIndex: 1,
    borderWidth: 1,
    borderColor: 'black',
    borderRadius: 0,
  },
  modelResults: {
    position:'absolute',
    left: 50,
    top: 100,
    width: 600/2,
    height: 800/2,
    zIndex: 20,
    borderWidth: 1,
    borderColor: 'black',
    borderRadius: 0,
  }
});
