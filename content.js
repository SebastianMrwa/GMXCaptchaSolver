function downloadImageScript() {
    console.log("Content Script triggered!");
    const captchaImage = document.querySelector('.captcha__image');
    if (captchaImage) {
      const imageUrl_before = captchaImage.getAttribute('src');
      const imageUrl = "https://interception1.gmx.net/logininterceptionfrontend/?" + imageUrl_before.split("?")[1];
  
      console.log(imageUrl);
  
      // Send a message to the background script
      chrome.runtime.sendMessage({ action: 'downloadImage', imageUrl });
    }
  }

// calls the classifyCaptcha python program
async function callPythonAPI() {
  //wait for a second for the image to be downloaded
  setTimeout(async function() {
    console.log('python-classification called');
    const apiUrl = 'http://127.0.0.1:5000/api/classifyCaptcha';

    const response = await fetch(apiUrl);

    const responseData = await response.json();
    console.log(responseData.result);
    const captchaText = responseData.result;

    //fill in the input field with the classified Captcha-Text
    let input = document.getElementById('0:form:captchaPanel:captchaImagePanel:captchaInput:topWrapper:inputWrapper:input');
    input.value = captchaText;
  }, 1000);
}
  
downloadImageScript();
callPythonAPI();