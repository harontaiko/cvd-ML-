const askForm = document.querySelector('.ask');
const askOutput = document.getElementById("ask-output");
const askInput = document.getElementById("qn");
let myListItem = document.querySelector('.predictions-list');

function convertYearsToDays(years) {
  const daysInYear = 365; // assume 365 days in a year
  return years * daysInYear;
}

const yearsInput = document.getElementById('years-input');
const daysOutput = document.getElementById('days-output');

yearsInput.addEventListener('input', () => {
  const years = Number(yearsInput.value);
  const days = convertYearsToDays(years);
  daysOutput.textContent = days;
});



askInput.addEventListener("keyup", function(event) {
  if (event.keyCode === 13) {
    event.preventDefault();
    // Trigger the form's submit event

    const question = askForm.elements["qn"].value;

    //send natural lang text to api,
    fetch("/ask", {
    method: "POST",
    body: JSON.stringify({
      text: question,
    }),
    headers: {
      "Content-Type": "application/json",
    },
  }).then((response) => response.json())
    .then((data) => {
      console.log(data);
      /*const result = data.predictiontext;
      //simulate typing animation(like real human in real time)
      let i = 0;
      const interval = setInterval(() => {
        askOutput.textContent += result.charAt(i);
        i++;
        if(i > result.length){
          clearInterval(interval);
        }
      }, 100);*/
      //askOutput.innerText = `Prediction: ${data.predictiontext}`;
    })

  }
});

document.addEventListener('DOMContentLoaded', ()=>{
    Object.keys(localStorage).forEach(function(key) {
      // Get the value of the current key
      const value = localStorage.getItem(key);
      myListItem.innerHTML += `<li class="w-full px-4 py-2 border-b border-gray-200 rounded-t-lg dark:border-gray-600">${key}: ${value}%</li>\n`;
  });
})

function validateInput(inputElement) {
  const inputValue = inputElement.value;
  const validValues = ["0", "1"];

  if (!validValues.includes(inputValue)) {
    inputElement.value = "";
    alert("Please enter only '0' or '1'");
  }
}

// Get references to the form and prediction output elements
const form = document.querySelector("#form");
const predictionOutput = document.getElementById("prediction-output");
const overlay = document.getElementById("overlay");
const plot = document.querySelector('#plot');

// Handle the form submit event
form.addEventListener("submit", (event) => {
  event.preventDefault();

  // Show the overlay
  overlay.style.display = "block";

  // Get the user inputs from the form
  const Age = form.elements["age"].value;
  const Gender = form.elements["gender"].value;
  const Height = form.elements["height"].value;
  const Weight = form.elements["weight"].value;
  const Systolic_bp = form.elements["systolic_bp"].value;
  const Diastolic_bp = form.elements["diastolic_bp"].value;
  const Cholesterol = form.elements["cholesterol"].value;
  const Glucose = form.elements["glucose"].value;
  const Smoke = form.elements["smoke"].value;
  const Alco = form.elements["alco"].value;
  const PhysicalActivity = form.elements["active"].value;

  // Make the prediction using the Flask API
  fetch("/", {
    method: "POST",
    body: JSON.stringify({
      age: Age,
      gender: Gender,
      height: Height,
      weight: Weight,
      systolic_bp: Systolic_bp,
      diastolic_bp: Diastolic_bp,
      cholesterol: Cholesterol,
      glucose: Glucose,
      smoke: Smoke,
      alco: Alco,
      active: PhysicalActivity,
    }),
    headers: {
      "Content-Type": "application/json",
    },
  })
    .then((response) => response.json())
    .then((data) => {
      // Hide the overlay after 3 seconds
      setTimeout(() => {
        overlay.style.display = "none";
        // Update the contents of the prediction output field
         //simulate typing animation(like real human in real time)
         result = `${data.prediction} with an Accuracy of ${data.accuracy}%`;
        let i = 0;
        const interval = setInterval(() => {
          predictionOutput.textContent += result.charAt(i);
          i++;
          if(i > result.length){
            clearInterval(interval);
          }
        }, 100);
        //predictionOutput.innerText = `Prediction: ${data.prediction}  | Accuracy: ${data.accuracy}%`;
        form.elements["prediction"].value = data.prediction;
        plot.src = 'data:image/png;base64,' + data.plot;
        console.log(data.prediction);
        console.log(data.accuracy);

        //store results on localstorage
        // Function to generate a random character
        function randomCharacter() {
          const characters = 'abcdefghijklmnopqrstuvwxyz0123456789';
          return characters[Math.floor(Math.random() * characters.length)];
        }

        // Function to generate a unique 4-character string
        function generateUniqueString() {
          let result = '';
          for (let i = 0; i < 4; i++) {
            result += randomCharacter();
          }
          return result;
        }

        const uniqLoc = generateUniqueString();
        localStorage.setItem(uniqLoc, `${data.prediction} (${data.accuracy})%`);

        const myKey = localStorage.getItem(uniqLoc);
        const myListItem = document.querySelector('.predictions-list');
        myListItem.innerHTML += `<li class="w-full px-4 py-2 border-b border-gray-200 rounded-t-lg dark:border-gray-600">${uniqLoc}:${myKey}</li>`;
      }, 2500);
    });
});

// Handle changes to the form inputs
form.addEventListener("input", () => {
  // Reset the prediction output and prediction input field
  predictionOutput.innerText = "";
  form.elements["prediction"].value = "";
});
