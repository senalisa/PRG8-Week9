const nn = ml5.neuralNetwork({ task: 'classification', debug: true })
nn.load('./model/model.json', modelLoaded)

function modelLoaded() {
console.log('Model Loaded!')

const form = document.getElementById('patientForm');
            form.addEventListener('submit', async event => {
            event.preventDefault();

            const Location = document.getElementById('Location').value;
            const Character = document.getElementById('Character').value;
            const Intensity = document.getElementById('Intensity').value;
            const Visual = document.getElementById('Visual').value;
            const Vertigo = document.getElementById('Vertigo').value;
            const Tinnitus = document.getElementById('Tinnitus').value;
            const Hypoacusis = document.getElementById('Hypoacusis').value;
            const Diplopia = document.getElementById('Diplopia').value;
            const Defect = document.getElementById('Defect').value;

            console.log({ Location, Character, Intensity, Visual, Vertigo, Tinnitus, Hypoacusis, Diplopia, Defect });

             // make the prediction using the neural network
            const results = await nn.predict({ Location, Character, Intensity, Visual, Vertigo, Tinnitus, Hypoacusis, Diplopia, Defect })

            console.log(results.length)

            });
}