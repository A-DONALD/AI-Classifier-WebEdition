let randomForestLabel = "earn";
let kNeighborsLabel = "earn";
let multinomialLabel = "earn";

// button to change the color of the background - switch theme
const checkbox = document.getElementById("checkbox")
checkbox.addEventListener("change", () => {
  document.body.classList.toggle("light")
})

// This function is responsible to print out the label of the document as a diagram with chart.js
function printResultDiagram(){
    let randomForestAccuracy = 85.5;
    let kNeighborsAccuracy = 83.3;
    let multinomialAccuracy = 79.8;

    let xValues, yValues;

    // config the different element of the diagram
    //if any element are the same, we merge them into the diagram and sum their percentages.
    if (randomForestLabel === kNeighborsLabel && kNeighborsLabel === multinomialLabel) {
        xValues = [randomForestLabel];
        yValues = [100];
    } else if (randomForestLabel === kNeighborsLabel) {
        xValues = [randomForestLabel, multinomialLabel];
        yValues = [randomForestAccuracy+kNeighborsAccuracy, multinomialAccuracy];
    } else if (kNeighborsLabel === multinomialLabel) {
        xValues = [kNeighborsLabel, randomForestLabel];
        yValues = [kNeighborsAccuracy+multinomialAccuracy, randomForestAccuracy];
    } else if (randomForestLabel === multinomialLabel) {
        xValues = [randomForestLabel, kNeighborsLabel];
        yValues = [randomForestAccuracy+multinomialAccuracy, kNeighborsAccuracy];
    } else {
        xValues = [randomForestLabel, kNeighborsLabel, multinomialLabel];
        yValues = [randomForestAccuracy, kNeighborsAccuracy, multinomialAccuracy];
    }

    // choose the color of the different element of the diagram
    const barColors = [
      "#87ceeb",
      "#000080",
      "#6495ed"
    ];

    // create a diagram to print out the solution with using chart.js
    new Chart("myChart", {
        type: "doughnut",
        data: {
        labels: xValues,
        datasets: [{
            backgroundColor: barColors,
            data: yValues
        }]
        },
        options: {
            title: {
                display: true,
                text: "Label of the text"
            }
        }
    });
}

document.addEventListener('DOMContentLoaded', function() {
    const textClassificationBtn = document.getElementById('text-classification');
    const docsClassificationBtn = document.getElementById('docs-classification');

    const textInput = document.getElementById('textInput');
    const resultDiv = document.getElementById('result-div');
    const dropZone = document.getElementById('dropZone');

    // Add event listeners for the drop zone
    dropZone.addEventListener('dragover', handleDragOver);

    // Add event listener for "docs-classification" button
    docsClassificationBtn.addEventListener('click', handleDocsClassification);

    // Function to manage the drag-and-drop hover event
    function handleDragOver(event) {
        event.preventDefault();
        event.stopPropagation();
        dropZone.classList.add('drag-over');
    }

    // Function to manage the drag-and-drop hover exit event
    function handleDragLeave(event) {
        event.preventDefault();
        event.stopPropagation();
        dropZone.classList.remove('drag-over');
    }

    // This function is responsible to classify the text in the textarea bag
    textClassificationBtn.addEventListener('click', function() {
        const text = textInput.value;
        const xhr = new XMLHttpRequest();

        // Make an AJAX request to the /classify route to obtain the label of the documents
        xhr.open('POST', '/txt-classify');
        xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
        xhr.onload = function() {
            if (xhr.status === 200) {
                const response = JSON.parse(xhr.responseText);
                const words = response.words;
                randomForestLabel = words[0];
                kNeighborsLabel = words[1];
                multinomialLabel = words[2];
                //print the result of each classifier in the "result-div"
                resultDiv.innerHTML = `
                    <p>${words[0]}</p>
                    <p>${words[1]}</p>
                    <p>${words[2]}</p>
                `;
                printResultDiagram();
            }
        };

        xhr.send('text=' + encodeURIComponent(text));
    });
});