var fs = require('fs');
var http = require('http');
var url = require('url');

var app = http.createServer(function(request, response) {
  var _url = request.url;
  var queryData = url.parse(_url, true).query;
  var pathName = url.parse(_url, true).pathname;

  if (pathName === '/') {
    if (queryData.id === undefined) {

    }
  } else {
    response.writeHead(404);
    response.end('Not Found');
  }
})

app.listen(3000);
