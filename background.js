// background script runs in the context of the gmx-site and downloads the Captcha
chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
    if (request.action === 'downloadImage' && request.imageUrl) {
      chrome.downloads.download({
        url: request.imageUrl,
        saveAs: false
      }, function(downloadId) {
        if (chrome.runtime.lastError) {
          console.error(chrome.runtime.lastError);
        } else {
          console.log('Captcha successfully downloaded!');
        }
      });
    }
  });
