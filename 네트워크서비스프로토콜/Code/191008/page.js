var http = require('http');
var fs = require('fs');
var url = require('url');

var app = http.createServer(function(request, response) {
  var _url = request.url;
  var queryData = url.parse(_url, true).query

  if (_url == "/") {
    title = 'Home';
  }
  if (_url == '/favicon.ico') {
    return response.writeHead(404);
  }

  response.writeHead(200);
  var template = `
  <!DOCTYPE html>
  <html>

  </html>`;
  response.end(template);
});

app.listen(3000);
