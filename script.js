
// Select canvas and buttons
const canvas = document.getElementById('drawCanvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;

async function loadModelAndPredict(input) {
    try {
        // Assuming the .json file is in your models folder
        console.log('Loading model');
        const model = await tf.loadLayersModel('funkar_men_it-våran/model.json');
        console.log(model.summary());

        console.log('Model loaded:', model);

        // Use the model for prediction or other tasks
        const inputTensor = tf.tensor(input);
        console.log(inputTensor)
        const prediction = model.predict(inputTensor);
        console.log("22");
        console.log('Prediction:', prediction);
    } catch (error) {
        console.error('Error::::', error);
    }
}

// Handle drawing on canvas
canvas.addEventListener('mousedown', () => {
    isDrawing = true;
    ctx.beginPath();
});

canvas.addEventListener('mousemove', (e) => {
    if (isDrawing) {
        ctx.lineWidth = 20;
        ctx.lineCap = 'round';
        ctx.strokeStyle = 'black';
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.stroke();
    }
});

canvas.addEventListener('mouseup', () => {
    isDrawing = false;
});

// Clear canvas button
document.getElementById('clearBtn').addEventListener('click', () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
});

// Placeholder for predict button (you can add the TensorFlow.js logic here)
document.getElementById('predictBtn').addEventListener('click', () => {
    const pixelData = getCanvasPixelData();

    // You can now pass `pixelData` to your TensorFlow.js model for predictions
    console.log(pixelData)
    const prediction = loadModelAndPredict(pixelData);
    console.log("after prdiction function" + prediction);
});


// Function to get the canvas as a 28x28 grayscale image and extract pixel values
function getCanvasPixelData() {
    // Create an offscreen canvas to resize the original canvas to 28x28
    const offscreenCanvas = document.createElement('canvas');
    offscreenCanvas.width = 28;
    offscreenCanvas.height = 28;
    const offscreenCtx = offscreenCanvas.getContext('2d');

    // Resize the 280x280 canvas image to 28x28
    offscreenCtx.drawImage(canvas, 0, 0, 280, 280, 0, 0, 28, 28);


    // Get the resized image data (28x28) from the offscreen canvas
    const imgData = offscreenCtx.getImageData(0, 0, 28, 28);

    // Prepare an array to hold the grayscale pixel values
    const grayscaleData = [];

    // Loop through the image data to convert it to grayscale
    for (let i = 0; i < imgData.data.length; i += 4) {
        // Extract the alpha channel
        const color = imgData.data[i+3]; 

        // Normalize the grayscale value to [0, 1]
        grayscaleData.push(color / 255);
    }

    // Return the grayscale pixel data (28 * 28 = 784 values)
    return grayscaleData;
}
