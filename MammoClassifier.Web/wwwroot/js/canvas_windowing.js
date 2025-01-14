function applyWindowing(canvas, windowWidth, windowCenter) {
    const ctx = canvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data; // Pixel data in [R, G, B, A, R, G, B, A, ...]

    const minPixelValue = windowCenter - 0.5 * windowWidth;
    const maxPixelValue = windowCenter + 0.5 * windowWidth;

    for (let i = 0; i < data.length; i += 4) {
        // Assuming the image is grayscale, R, G, and B will have the same values.
        const intensity = data[i]; // Red channel (grayscale image)

        // Apply windowing
        let newIntensity = 255 * ((intensity - minPixelValue) / (maxPixelValue - minPixelValue));
        newIntensity = Math.max(0, Math.min(255, newIntensity)); // Clamp to 0-255

        // Update R, G, and B channels
        data[i] = newIntensity;     // Red
        data[i + 1] = newIntensity; // Green
        data[i + 2] = newIntensity; // Blue
        // Alpha (data[i + 3]) remains unchanged
    }

    // Put the adjusted image data back on the canvas
    ctx.putImageData(imageData, 0, 0);
}


function contrastImage(imgData, contrast) {  //input range [-100..100]
    var d = imgData.data;
    contrast = (contrast / 100) + 1;  //convert to decimal & shift range: [0..2]
    var intercept = 128 * (1 - contrast);
    for (var i = 0; i < d.length; i += 4) {   //r,g,b,a
        d[i] = d[i] * contrast + intercept;
        d[i + 1] = d[i + 1] * contrast + intercept;
        d[i + 2] = d[i + 2] * contrast + intercept;
    }
    return imgData;
}