var fs = require('fs');
var template = require('./lib/template.js');
var express = require('express');
var bodyParser = require('body-parser');
var compression = require('compression');
var topicRouter = require('./routes/topic.js');
var indexRouter = require('./routes/index.js');

const app = express()

app.use(express.static('public'));
app.use(bodyParser.urlencoded({extended: false}))
app.use(compression());

app.get('*', function(req, res, next) {
  fs.readdir('./data', function(error, filelist) {
    req.list = filelist;
    next();
  })
})

app.use('/', indexRouter)
app.use('/topic', topicRouter)

app.use(function(req, res, next) {
    res.status(404).send('Sorry cant find that!');
});

app.use(function(err, req, res, next) {
  console.error(err.stack)
  res.status(500).send('Something broke!')
})

app.listen(3000, () => console.log('Example app listening on port 3000!'))
