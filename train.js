const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');
const mnist = require("mnist");

class MnistData {
    constructor() {
      this.trainBatchIndex = 0;
      this.testBatchIndex = 0;
    }
  
    async load() {
        let set = mnist.set(8000, 2000);

        this.trainingSet = set.training;
        this.testSet = set.test;


    }
    
    nextTrainBatch(batchSize) {
      let batch = this.trainingSet.slice(this.trainBatchIndex, batchSize)
      this.trainBatchIndex += batchSize
      
      return batch
    }
  
    nextTestBatch(batchSize) {
      let batch = this.testSet.slice(this.testBatchIndex, batchSize)
      this.testBatchIndex += batchSize
      return batch
    }
}

function getModel() {
    const model = tf.sequential();
    
    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const IMAGE_CHANNELS = 1;  
    
    // In the first layer of our convolutional neural network we have 
    // to specify the input shape. Then we specify some parameters for 
    // the convolution operation that takes place in this layer.
    model.add(tf.layers.conv2d({
      inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
      kernelSize: 5,
      filters: 8,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    }));
  
    // The MaxPooling layer acts as a sort of downsampling using max values
    // in a region instead of averaging.  
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    
    // Repeat another conv2d + maxPooling stack. 
    // Note that we have more filters in the convolution.
    model.add(tf.layers.conv2d({
      kernelSize: 5,
      filters: 16,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    
    // Now we flatten the output from the 2D filters into a 1D vector to prepare
    // it for input into our last layer. This is common practice when feeding
    // higher dimensional data to a final classification output layer.
    model.add(tf.layers.flatten());
  
    // Our last layer is a dense layer which has 10 output units, one for each
    // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
    const NUM_OUTPUT_CLASSES = 10;
    model.add(tf.layers.dense({
      units: NUM_OUTPUT_CLASSES,
      kernelInitializer: 'varianceScaling',
      activation: 'softmax'
    }));
  
    
    // Choose an optimizer, loss function and accuracy metric,
    // then compile and return the model
    const optimizer = tf.train.adam();
    model.compile({
      optimizer: optimizer,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });
  
    return model;
}

async function train(model, data) {
    const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
    const container = {
      name: 'Model Training', tab: 'Model', styles: { height: '1000px' }
    };
    
    const BATCH_SIZE = 256;
    const TRAIN_DATA_SIZE = data.trainingSet.length;
    const TEST_DATA_SIZE = data.testSet.length;
  
    const [trainXs, trainYs] = tf.tidy(() => {
      const d = data.trainingSet;
      return [
        tf.tensor(d.map((x) => x.input)).reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
        tf.tensor(d.map((y) => y.output))
      ];
    });
  
    const [testXs, testYs] = tf.tidy(() => {
      const d = data.nextTestBatch(TEST_DATA_SIZE);
      return [
        tf.tensor(d.map((x) => x.input)).reshape([TEST_DATA_SIZE, 28, 28, 1]),
        tf.tensor(d.map((y) => y.output))
      ];
    });
  
    console.log(trainXs)
    console.log(trainYs)

    function onBatchEnd(batch, logs) {
      console.log("logs: " + JSON.stringify(logs, null, 4));
    }

    return model.fit(trainXs, trainYs, {
      batchSize: BATCH_SIZE,
      validationData: [testXs, testYs],
      epochs: 10,
      shuffle: true,
      callbacks: {onBatchEnd}
    }).then(info => {
      console.log('Accuracy', info.history.acc);
      const saveResult = model.save('file://./saved_model/statedict');
    });
}

const model = getModel();
let data = new MnistData()
data.load()

async function run(model, data){
  await train(model, data)
}

run(model, data)