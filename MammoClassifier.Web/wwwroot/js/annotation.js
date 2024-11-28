function onImageRendered(e) {
    const eventData = e.detail;
    // set the canvas context to the image coordinate system
    cornerstone.setToPixelCoordinateSystem(eventData.enabledElement, eventData.canvasContext);

    const context = eventData.canvasContext;

    // style the context
    context.strokeStyle = "green";
    context.lineWidth = 1 / eventData.viewport.scale;

    // calculate where the canvas is on the window
    // (used to help calculate mouseX/mouseY)
    var $canvas = $("#mapImage1 > canvas");
    var canvasOffset = $canvas.offset();
    var offsetX = canvasOffset.left;
    var offsetY = canvasOffset.top;
    var scrollX = $canvas.scrollLeft();
    var scrollY = $canvas.scrollTop();

    // this flage is true when the user is dragging the mouse
    var isDown = false;

    // these vars will hold the starting mouse position
    var startX;
    var startY;

    var prevStartX = 0;
    var prevStartY = 0;

    var prevWidth = 0;
    var prevHeight = 0;

    function handleMouseDown(e) {
        e.preventDefault();
        e.stopPropagation();

        // save the starting x/y of the rectangle
        startX = parseInt(e.clientX - offsetX);
        startY = parseInt(e.clientY - offsetY);

        // set a flag indicating the drag has begun
        isDown = true;
    }

    function handleMouseUp(e) {
        e.preventDefault();
        e.stopPropagation();

        // the drag is over, clear the dragging flag
        isDown = false;
        context.strokeRect(prevStartX, prevStartY, prevWidth, prevHeight);
    }

    function handleMouseOut(e) {
        e.preventDefault();
        e.stopPropagation();

        // the drag is over, clear the dragging flag
        isDown = false;
    }

    function handleMouseMove(e) {
        e.preventDefault();
        e.stopPropagation();

        // if we're not dragging, just return
        if (!isDown) {
            return;
        }

        // get the current mouse position
        mouseX = parseInt(e.clientX - offsetX);
        mouseY = parseInt(e.clientY - offsetY);

        // Put your mousemove stuff here



        // calculate the rectangle width/height based
        // on starting vs current mouse position
        var width = mouseX - startX;
        var height = mouseY - startY;

        // clear the canvas
        context.clearRect(0, 0, $canvas.width, $canvas.height);

        // draw a new rect from the start position 
        // to the current mouse position
        context.strokeRect(startX, startY, width, height);

        prevStartX = startX;
        prevStartY = startY;

        prevWidth = width;
        prevHeight = height;
    }


    // listen for mouse events
    $canvas.mousedown(function (e) {
        handleMouseDown(e);
    });
    $canvas.mousemove(function (e) {
        handleMouseMove(e);
    });
    $canvas.mouseup(function (e) {
        handleMouseUp(e);
    });
    $canvas.mouseout(function (e) {
        handleMouseOut(e);
    });

    //document.getElementById('topright').textContent = "Render Time:" + eventData.renderTimeInMs + " ms";
    //document.getElementById('bottomleft').textContent = "WW/WL:" + Math.round(eventData.viewport.voi.windowWidth) + "/" + Math.round(eventData.viewport.voi.windowCenter);
    //document.getElementById('bottomright').textContent = "Zoom:" + eventData.viewport.scale.toFixed(2);

}

element_cc_dcm.addEventListener('cornerstoneimagerendered', onImageRendered);




