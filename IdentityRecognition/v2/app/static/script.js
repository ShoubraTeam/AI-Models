function previewImage(inputId, previewId) {
    const input = document.getElementById(inputId);
    const preview = document.getElementById(previewId);

    input.addEventListener("change", () => {
        const file = input.files[0];

        if (file) {
            preview.src = URL.createObjectURL(file);
            preview.style.display = "block";
        }
    });
}

previewImage("image1", "preview1");
previewImage("image2", "preview2");

async function verifyFaces() {
    const image1 = document.getElementById("image1").files[0];
    const image2 = document.getElementById("image2").files[0];
    const resultDiv = document.getElementById("result");
    const loader = document.getElementById("loader");

    if (!image1 || !image2) {
        alert("Please upload both images.");
        return;
    }

    resultDiv.className = "result hidden";
    loader.classList.remove("hidden");

    const formData = new FormData();
    formData.append("image1", image1);
    formData.append("image2", image2);

    try {
        const response = await fetch("/verify", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        loader.classList.add("hidden");
        resultDiv.classList.remove("hidden");

        if (data.verified) {
            resultDiv.className = "result same";
            resultDiv.innerHTML = `
                ✅ Same Person<br>
                Similarity: ${data.similarity}<br>
                Threshold: ${data.threshold}
            `;
        } else {
            resultDiv.className = "result different";
            resultDiv.innerHTML = `
                ❌ Different Persons<br>
                Similarity: ${data.similarity}<br>
                Threshold: ${data.threshold}
            `;
        }

    } catch (error) {
        loader.classList.add("hidden");
        resultDiv.className = "result different";
        resultDiv.innerHTML = "Error while verifying faces.";
        console.error(error);
    }
}