var express = require('express');
var template = require('../lib/template.js');
var fs = require('fs');

var router = express.Router();

router.get('/', function(req, res) {
  var title = 'Welcome';
  var description = 'Hello, Node.js';
  var list = template.list(req.list);
  var html = template.HTML(title, list,
    `<h2>${title}</h2>${description}`,
    `<a href="/topic/create">create</a>
      <img src="/images/test.jpg">`
  );

  res.send(html);
});

module.exports = router
