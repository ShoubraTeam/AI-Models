const jobPostBtn = document.getElementById("jobPostBtn");
const jobTitleIP = document.getElementById("jobTitleInput");
const jobDescIP = document.getElementById("jobDescInput");
const toolsBtns = document.getElementById("toolsBtns");
const toolsSubmitBtn = document.getElementById("toolsSubmitBtn");
const orgDescDiv = document.querySelector(".original-job-desc");
const enhDescDiv = document.querySelector(".enhanced-job-desc");

jobPostBtn.addEventListener("click", async e => {
    // cfg
    e.preventDefault();
    orgDescDiv.textContent = "";
    enhDescDiv.textContent = "";
    toolsBtns.innerHTML = "";
    toolsSubmitBtn.style.cssText = 'display: none';

    const title = jobTitleIP.value;
    const desc = jobDescIP.value;

    const payload = {
        job_title : title,
        job_desc  : desc
    };


    const request = await fetch("/detect_tools", {
        method  : "POST",
        headers : {"Content-Type" : "application/json"},
        body    : JSON.stringify(payload) 
    });

    const response = await request.json();
    if (response['status'] == 'success') {
        if (response['has_tools'] === 1) {
            enhanceJobRequest("/enhance_desc", title, desc, response['suggested_tools']);
        } else {
            // show the suggested tools to the user
            const suggestedTools = Array.from(response['suggested_tools']);
            const userPreferredTools = Array();
            suggestedTools.forEach(tool => {
                const btn = document.createElement('button');
                btn.textContent = tool;
                toolsBtns.appendChild(btn);
                

                btn.addEventListener('click', e => {
                    e.preventDefault();
                    userPreferredTools.push(tool);
                    e.target.classList.add('selected');
                });
            });

            // show the select btn
            toolsSubmitBtn.style.cssText = 'display: block';
            
            toolsSubmitBtn.addEventListener('click', e => {
                e.preventDefault();
                enhanceJobRequest("/enhance_desc", title, desc, userPreferredTools);
            });
        }
    }
});

async function enhanceJobRequest(endpoint, job_title, job_desc, tools) {
    let payload;
    if (tools === null) {
        payload = {
            job_title : job_title,
            job_desc  : job_desc,
        };
    } else {
        payload = {
            job_title : job_title,
            job_desc  : job_desc,
            tools     : tools
        };
    }
    

    
    const request = await fetch(endpoint, {
        method  : "POST",
        headers : {"Content-Type" : "application/json"},
        body    : JSON.stringify(payload) 
    });

    const response = await request.json();
    if (response['status'] === 'success') {
        orgDescDiv.textContent = response['original_desc'];
        enhDescDiv.textContent = response['enhanced_desc']
    }
};