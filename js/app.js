const criminalForm = document.getElementById("criminalForm");
const theName = document.getElementById("name");
const conviction = document.getElementById("conviction");
const totalConvictions = document.getElementById("totalConvictions");
const result = document.getElementById("result");

let lastSubmissionTime = 0;
const cooldownTime = 10000;

var lastRequest = {
    name: null,
    conviction: null,
    totalNumOfConvictions: null,
};

async function predictSentence(name, conviction, totalNumOfConvictions) {
    const response = await fetch("http://localhost:5000/process", {
        method: "POST",
        mode: "cors",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            name,
            conviction,
            totalNumOfConvictions,
        }),
    });

    const data = await response.json();
    return data;
}

document.querySelector("form").addEventListener("submit", function (event) {
    console.log("Form submitted without reloading");
});

async function getData(theName, convict, totalConvict) {
    const data = await predictSentence(theName, convict, totalConvict);
    return data;
}

criminalForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const currentTime = Date.now();
    if (currentTime - lastSubmissionTime < cooldownTime) {
        alert("Please wait 10 seconds before submitting again.");
        return;
    }
    lastSubmissionTime = currentTime;
    const nameValue = theName.value;
    const convictionValue = conviction.value;
    const totalConvictionsValue = totalConvictions.value;
    lastRequest.name = nameValue;
    lastRequest.conviction = convictionValue;
    lastRequest.totalNumOfConvictions = totalConvictionsValue;
    var ultData = await getData(
        nameValue,
        convictionValue,
        totalConvictionsValue
    );
    var res = (resultData = await JSON.parse(ultData));
    console.log(res);
    result.textContent = `${nameValue}`;
    result.textContent = `The convict ${nameValue} has been sentenced to ${resultData["Predicted Sentence to be Served: "]} years in prison, and has been fined £${resultData["Predicted Amount to Fine: "]} for the crime of ${convictionValue}`;
});
