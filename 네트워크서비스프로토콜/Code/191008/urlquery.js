var http = require('http');
var fs = require('fs');
var url = require('url');

var app = http.createServer(function(req, res) {
  var _url = req.url;
  var queryData = url.parse(_url, true).query;

  // console.log(queryData);

  if (_url == "/") {
    _url = res.end('Welcome');
  }
  if (_url = "/favicon.ico") {
    return res.writeHead(404);
  }

  res.writeHead(200);
  res.end(queryData.id);
});

app.listen(3000);
