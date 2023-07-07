let randomForestLabel;
let kNeighborsLabel;
let multinomialLabel;

document.addEventListener('DOMContentLoaded', function() {
    const textClassificationBtn = document.getElementById('text-classification');
    const textInput = document.getElementById('textInput');
    const resultDiv = document.getElementById('result-div');


    textClassificationBtn.addEventListener('click', function() {
        const text = textInput.value;

        // Effectuer une requête AJAX vers la route /classify pour obtenir les mots
        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/txt-classify');
        xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
        xhr.onload = function() {
            if (xhr.status === 200) {
                const response = JSON.parse(xhr.responseText);
                const words = response.words;
                randomForestLabel = words[0];
                kNeighborsLabel = words[1];
                multinomialLabel = words[2];
                resultDiv.innerHTML = `
                    <p>${words[0]}</p>
                    <p>${words[1]}</p>
                    <p>${words[2]}</p>
                `;

            }
        };
        xhr.send('text=' + encodeURIComponent(text));
    });
});

let randomForestAccuracy = 85.5;
let kNeighborsAccuracy = 83.3;
let multinomialAccuracy = 79.8;

let xValues, yValues;

// config the different element of the diagram
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

// button to change the color of the background - change theme
const checkbox = document.getElementById("checkbox")
checkbox.addEventListener("change", () => {
  document.body.classList.toggle("light")
})