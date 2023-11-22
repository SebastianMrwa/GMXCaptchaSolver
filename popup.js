const errorMessage = document.getElementById('errorMessage');
// Returns true/false if the URL matches
function isRightUrl(callback) {
    chrome.tabs.query({ active: true, currentWindow: true }, function(tabs){
        const activeTab = tabs[0];
        const url = activeTab.url;
        const targetURL = 'https://interception1.gmx.net/';
        callback(url.includes(targetURL));
    });
}

async function downloadCaptcha() {
    let [tab] = await chrome.tabs.query({ active: true });
    //get the current URL
    isRightUrl(function(isRight) {
        //check if the current URL is right
        if(isRight){
            //call the content.js file
            chrome.scripting.executeScript({
                target: {tabId: tab.id},
                files: ['content.js']
            });
        } else {
            errorMessage.textContent = 'Error: Wrong site.';
        }
    });
}

async function callPythonAPI() {
    const apiUrl = 'http://127.0.0.1:5000/api/classifyCaptcha';

    const response = await fetch(apiUrl);

    const responseData = await response.json();
    console.log(responseData.result);
}

document.getElementById("solveButton").addEventListener("click", downloadCaptcha);