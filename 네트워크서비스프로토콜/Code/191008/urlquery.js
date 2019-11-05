var http = require('http');
var fs = require('fs');
var url = require('url');

var app = http.createServer((req, res) => {
  var _url = req.url; // XXX:
  var queryData = url.parse(_url, true).query;

  console.log(_url, queryData);

  // console.log(queryData);

  if (_url == "/") {
    _url = res.end('Welcome');
  }
  // else if (_url = "/favicon.ico") {
  //   return res.writeHead(404);
  // }

  res.writeHead(200);
  res.end(queryData.id);
});

app.listen(3000);
