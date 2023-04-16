let nn, trainData, testData

function loadData() {
    Papa.parse("./data/data.csv", {
        download: true,
        header: true, 
        dynamicTyping: true,
        complete: (results) => cleanData(results.data)
    })
}

function cleanData(data) {
    // console.table(data)

    // haal de data uit de CSV die je nodig hebt, inclusief het label waarop je wil trainen
    // met de filter functie checken we dat de traindata uit nummers bestaat
    const cleanData = data
        .map(patient => ({
            Location: patient.Location,
            Character: patient.Character,
            Intensity: patient.Intensity,
            Visual: patient.Visual,
            Vertigo: patient.Vertigo,
            Tinnitus: patient.Tinnitus,
            Hypoacusis: patient.Hypoacusis,
            Diplopia: patient.Diplopia,
            Defect: patient.Defect,
            Type: patient.Type
        }))
        .filter(patient =>
            typeof patient.Location === "number" &&
            typeof patient.Character === "number" &&
            typeof patient.Intensity === "number" &&
            typeof patient.Visual === "number" &&
            typeof patient.Vertigo === "number" &&
            typeof patient.Tinnitus === "number" &&
            typeof patient.Hypoacusis === "number" &&
            typeof patient.Diplopia === "number" &&
            typeof patient.Defect === "number" 
        )

    cleanData.sort(() => (Math.random() - 0.5))
    trainData = cleanData.slice(0, Math.floor(data.length * 0.75))
    testData = cleanData.slice(Math.floor(data.length * 0.75) + 1)

    createNeuralNetwork(trainData)
}

function createNeuralNetwork(data) {
    nn = ml5.neuralNetwork({ task: 'classification', debug: true })

    const options = { 
        task: 'classification', 
        debug: true,
        layers: [
            {
                type: 'dense',
                units: 32,
                activation: 'relu',
            }, 
            {
                type: 'dense',
                activation: 'softmax',
            },
        ]
    }
    nn = ml5.neuralNetwork(options)

    for (let patient of data) {
        const inputs = { 
            Location: patient.Location,
            Character: patient.Character,
            Intensity: patient.Intensity,
            Visual: patient.Visual,
            Vertigo: patient.Vertigo,
            Tinnitus: patient.Tinnitus,
            Hypoacusis: patient.Hypoacusis,
            Diplopia: patient.Diplopia,
            Defect: patient.Defect,
        }
        const output = { 
            Type: patient.Type 
        } 
        nn.addData(inputs, output)
    }

    nn.normalizeData()
    nn.train({ epochs: 60 }, () => getAccuracy())

}


async function getAccuracy() {
    let correctPredictions = 0

    for (let patient of testData) {
        const inputs = {
            Location: patient.Location,
            Character: patient.Character,
            Intensity: patient.Intensity,
            Visual: patient.Visual,
            Vertigo: patient.Vertigo,
            Tinnitus: patient.Tinnitus,
            Hypoacusis: patient.Hypoacusis,
            Diplopia: patient.Diplopia,
            Defect: patient.Defect,
        }
        const result = await nn.classify(inputs)
        console.log(`Predicted: ${result[0].label}, Actual data: ${patient.Type}`)
        if (result[0].label === patient.Type) {
            correctPredictions++
        }
    }

    console.log(`Correcte voorspellingen ${correctPredictions} van de ${testData.length}, dit is ${((correctPredictions / testData.length) * 100).toFixed(2)} %`)
}

// Add event listener to the save button
const saveBtn = document.getElementById("save-btn");
saveBtn.addEventListener("click", () => {
  nn.save();
  console.log("Model saved!");
});

loadData()