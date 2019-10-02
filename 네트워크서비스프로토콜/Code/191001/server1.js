const http = require('http');

const onRequest = (req, res) => {
  console.log('request received');
  res.end('<p1> Hello Node </p1>');
}

http.createServer((req, res) => {
    console.log("???");
    req.onRequest()
}).listen(8080, () => {
  
})
