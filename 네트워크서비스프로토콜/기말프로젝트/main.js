var http = require('http');
var mysql = require('mysql');
var pw = require('./pw.js');
var fs = require('fs');
var url = require('url');
var path = require('path');
var template = require('./lib/template.js');
var async = require('async');

var db = mysql.createConnection({
  host : 'localhost',
  user : 'root',
  password : pw.pw,
  database : 'travel_schedule'
});
db.connect();

var app = http.createServer(function(request,response) {
  var _url = request.url;
  var queryData = url.parse(_url, true).query;
  var pathname = url.parse(_url, true).pathname;

  if (pathname === '/') {
    if (queryData.id === undefined) {
      db.query('SELECT * FROM schedule', function(error, schedules) {
        var schedule_list = template.schedule_list(schedules);
        var review_list = template.review_list(schedules);
        var html = template.HTML(schedule_list, review_list, '','');
        response.writeHead(200, {'Content-Type': 'text/html'});
        response.end(html);
      })
    }
  } else if (pathname == '/schedule') {
    db.query(`SELECT * FROM consists WHERE SCHEDULE_ID = ${queryData.id}`, function(err,consists) {
      if (err) {
        // 없는 schedule id
        response.writeHead(404);
        response.end('<h1>404 NOT FOUND</h1>')
      } else {
        // console.log(consists);
        var body = '';
        db.query(`SELECT * FROM schedule WHERE SCHEDULE_ID = ${queryData.id}`, function(err2,schedule) {
          if (err2) {
            throw err2;
          }
          body += `<div class='schedule_name'>${schedule[0].SCHEDULE_NAME}</div>
                   <div class='schedule_description'>✈️${schedule[0].SCHEDULE_DESCRIPTION}</div>`;
        })

        async.waterfall([
          function(callback) {
            for (var i=0; i<consists.length; i++) {
              var day = consists[i].CONSISTS_DAY;
              var time = consists[i].CONSISTS_TIME;
              db.query(`SELECT * FROM activity WHERE ACTIVITY_ID = ${consists[i].ACTIVITY_ID}`, function(err3, activity) {
                var activity_name = activity[0].ACTIVITY_NAME;
                var activity_description = activity[0].ACTIVITY_DESCRIPTION;
                var activity_image = activity[0].ACTIVITY_IMAGE;
                // let place_name;
                db.query(`SELECT * FROM place WHERE PLACE_ID = ${activity[0].PLACE_ID}`,function(err4, place) {
                  var place_name = place[0].PLACE_NAME;

                  body += `
                    <div class="activity">
                    <div class="text_section">
                    <div class="activity_time">${day}일차 ${time}</div>
                    <div class="place_name">${place_name}</div>
                    <div class="activity_name">${activity_name}</div>
                    <div class="activity_description">${activity_description}</div></div>`
                  if (activity_image != null) {
                    body += `<div class="activity_image"><img src=${activity_image}></div></div>`
                  } else {
                    body += `<div class="activity_image"></div></div>`
                  }
                  console.log(i);
                  console.log(body);
                  if (i== consists.length-1) {
                    callback(null, body);
                  }
                });
              })
            }
          }
        ],
        function (err, body) {
          db.query('SELECT * FROM schedule', function(error, schedules) {

            var schedule_list = template.schedule_list(schedules);
            var review_list = template.review_list(schedules);
            var html = template.HTML(schedule_list, review_list, '', body);
            response.writeHead(200, {'Content-Type': 'text/html'});
            response.end(html);
          })
        });

      }
    })
  } else if (pathname == '/review') {

  } else if (pathname == '/images') {
    fs.readFile(`./images/${queryData.image}`, function(image_error, data){
      if (image_error) {
        throw image_error;
      }
      response.writeHead(200, {'Content-Type' : 'image/jpeg'});
      response.end(data);
    });
  } else if (pathname == '/css') {
    // console.log(queryData.css)
    fs.readFile(`./css/${queryData.css}`, function(css_error, data){
      if (css_error) {
        throw css_error;
      }
      response.writeHead(200, {'Content-Type' : 'text/css'});
      response.end(data);
    });
  } else if (pathname == '/fonts') {
    fs.readFile(`./fonts/${queryData.font}`, function(font_error, data) {
      if (font_error) {
        throw font_error;
      }
      response.writeHead(200, {'Content-Type' : 'aplication/font-sfnt'});
      response.end(data);
    })
  } else {
    response.writeHead(404);
    response.end('<h1>404 NOT FOUND</h1>')
  }
})

app.listen(3000);
