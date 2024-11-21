// image enable the element
const element1 = document.getElementById("dicomImage1");
const element2 = document.getElementById("dicomImage2");
const element1_map = document.getElementById("mapImage1");
const element2_map = document.getElementById("mapImage2");

var element1_org_viewport = null;
var element2_org_viewport = null;

function setViewportForElement(element) {
  let viewport = cornerstone.getViewport(element);
  viewport.voi.windowWidth = parseFloat(document.getElementById("ww").value);
  viewport.voi.windowCenter = parseFloat(document.getElementById("wc").value);
  cornerstone.setViewport(element, viewport);
}

function invertElement(element) {
  let viewport = cornerstone.getViewport(element);
  viewport.invert = !viewport.invert;
  cornerstone.setViewport(element, viewport);
}

// Add event handler to the ww/wc apply button
document.getElementById("apply").addEventListener("click", function (e) {
  setViewportForElement(element1);
  setViewportForElement(element2);
});

document.getElementById("invert").addEventListener("click", function (e) {
  invertElement(element1);
  invertElement(element2);
});

document.getElementById("reset").addEventListener("click", function (e) {
  cornerstone.reset(element1);
  cornerstone.reset(element2);
});

// add event handlers to mouse move to adjust window/center
function addMouseEvents(element, overlay) {
    overlay.addEventListener("mousedown", function (e) {
    let lastX = e.pageX;
    let lastY = e.pageY;

    function mouseMoveHandler(e) {
      const deltaX = e.pageX - lastX;
      const deltaY = e.pageY - lastY;
      lastX = e.pageX;
      lastY = e.pageY;

      let viewport = cornerstone.getViewport(element);
      viewport.voi.windowWidth += deltaX / viewport.scale;
      viewport.voi.windowCenter += deltaY / viewport.scale;
      cornerstone.setViewport(element, viewport);
    }

    function mouseUpHandler() {
      document.removeEventListener("mousemove", mouseMoveHandler);
      document.removeEventListener("mouseup", mouseUpHandler);
    }

    document.addEventListener("mousemove", mouseMoveHandler);
    document.addEventListener("mouseup", mouseUpHandler);
  });
}

addMouseEvents(element1, element1_map);
addMouseEvents(element2, element2_map);
