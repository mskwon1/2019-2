var http = require('http');
var mysql = require('mysql');
var pw = require('./pw.js');
var fs = require('fs');
var qs = require('querystring');
var url = require('url');
var path = require('path');
var template = require('./lib/template.js');
var async = require('async');
var multiparty = require('multiparty');
var util = require('util');

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
      var body = '';
      db.query(`SELECT * FROM schedule WHERE SCHEDULE_ID = ${queryData.id}`, function(err2,schedule) {
        var schedule_name = schedule[0].SCHEDULE_NAME
        var schedule_description = schedule[0].SCHEDULE_DESCRIPTION
        var schedule_country = schedule[0].SCHEDULE_COUNTRY
        db.query(`SELECT * FROM consists JOIN activity USING (ACTIVITY_ID) JOIN place USING (PLACE_ID) WHERE SCHEDULE_ID = ${queryData.id}
                      ORDER BY CONSISTS_DAY, CONSISTS_TIME`, function(err,consists) {
          let promise = new Promise(function(resolve, reject) {
            body += addBody(body, consists, schedule_name, schedule_description, queryData.id)
            resolve(body)
          })

          function addBody(body, consists, schedule_name, schedule_description, id) {
            body += `<div class='schedule_name'>[${schedule_country}]${schedule_name}</div>
                      <div class='schedule_description'>✈️${schedule_description}</div>
                      <hr>
                      <div class='controls'>
                        <a href='/add_consist?id=${id}'>일정 추가하기</a><br>
                        <a href='/update_schedule?id=${id}'>일정 수정하기</a>
                      </div>
                      `;
            for (var i=0; i<consists.length; i++) {
              var day = consists[i].CONSISTS_DAY;
              var time = consists[i].CONSISTS_TIME;
              var activity_name = consists[i].ACTIVITY_NAME;
              var activity_description = consists[i].ACTIVITY_DESCRIPTION;
              var activity_image = consists[i].ACTIVITY_IMAGE;
              var place_name = consists[i].PLACE_NAME;

              body += `
                <div class="activity">
                <div class="text_section">
                <div class="activity_time">${day}일차 🕒 ${time}</div>
                <div class="place_name">${place_name}</div>
                <div class="activity_name">${activity_name}</div>
                <div class="activity_description">${activity_description}</div></div>`

              if (activity_image != null) {
                body += `<div class="activity_image"><img src=${activity_image}></div></div>`
              } else {
                body += `<div class="activity_image"></div></div>`
              }

            }

            return body
          }

          promise.then(function(contents) {
            db.query('SELECT * FROM schedule', function(error, schedules) {
              var schedule_list = template.schedule_list(schedules);
              var review_list = template.review_list(schedules);
              var html = template.HTML(schedule_list, review_list, '', body);
              response.writeHead(200, {'Content-Type': 'text/html'});
              response.end(html);
            })
          })
        })
      })
  } else if (pathname == '/create_schedule') {
      db.query('SELECT * FROM schedule', function(error, schedules) {
        var body = `
        <form action="/create_schedule_process" method="post">
          <p>이름</p>
          <p><input type="text" name="name" placeholder="Schedule Name"></p>
          <p>설명</p>
          <p><textarea name="description" placeholder="Schedule Description"></textarea></p>
          <p>나라이름</p>
          <p><input type="text" name="country" placeholder="Schedule Country"></p>
          <p><input type="submit" value="저장"></p>
        </form>
        `
        var schedule_list = template.schedule_list(schedules);
        var review_list = template.review_list(schedules);
        var html = template.HTML(schedule_list, review_list, '', body);
        response.writeHead(200, {'Content-Type': 'text/html'});
        response.end(html);
      })
  } else if (pathname == '/create_schedule_process') {
      var body = '';
      request.on('data', function(data) {
        body = body + data;
      })
      request.on('end', function() {
        var post = qs.parse(body);
        var name = post.name;
        var description  = post.description;
        var country = post.country
        if (name != undefined && description != undefined &&
            name.length <= 45 && description.length <= 90) {
          db.query(`INSERT INTO schedule (SCHEDULE_NAME, SCHEDULE_DESCRIPTION, SCHEDULE_COUNTRY)
                    VALUES(?,?,?)`, [name, description,country], function(err, result) {
            if (err) {
              throw err;
            }
            response.writeHead(302, {Location:`/schedule?id=${result.insertId}`})
            response.end()
          })
        } else {
          alert('이름은 45자 이하, 설명은 90자이하여야합니다')
          response.writeHead(302, {Location:'/create_schedule'})
          response.end()
        }
      })
  } else if (pathname == '/update_schedule') {
      var body = '';
      db.query(`SELECT * FROM schedule WHERE SCHEDULE_ID = ${queryData.id}`, function(err2,schedule) {
        var schedule_name = schedule[0].SCHEDULE_NAME
        var schedule_description = schedule[0].SCHEDULE_DESCRIPTION
        var schedule_country = schedule[0].SCHEDULE_COUNTRY
        db.query(`SELECT * FROM consists JOIN activity USING (ACTIVITY_ID) JOIN place USING (PLACE_ID) WHERE SCHEDULE_ID = ${queryData.id}
                      ORDER BY CONSISTS_DAY, CONSISTS_TIME`, function(err,consists) {
          let promise = new Promise(function(resolve, reject) {
            body += addBody(body, consists, schedule_name, schedule_description, queryData.id)
            resolve(body)
          })

          function addBody(body, consists, schedule_name, schedule_description, id) {
            body += `<div class='schedule_name'>[${schedule_country}]${schedule_name}</div>
                      <div class='schedule_description'>✈️${schedule_description}</div>
                      <hr>
                      `;
            for (var i=0; i<consists.length; i++) {
              var day = consists[i].CONSISTS_DAY;
              var time = consists[i].CONSISTS_TIME;
              var activity_id = consists[i].ACTIVITY_ID;
              var activity_name = consists[i].ACTIVITY_NAME;
              var activity_description = consists[i].ACTIVITY_DESCRIPTION;
              var activity_image = consists[i].ACTIVITY_IMAGE;
              var place_name = consists[i].PLACE_NAME;

              body += `
                <div class="activity">
                <div class="text_section">
                <div class="activity_time">${day}일차 🕒 ${time}</div>
                <div class="place_name">${place_name}</div>
                <div class="activity_name">${activity_name}</div>
                <div class="activity_description">${activity_description}</div></div>`

              if (activity_image != null) {
                body += `<div class="activity_image"><img src=${activity_image}></div>`
              } else {
                body += `<div class="activity_image"></div>`
              }

              body += `
                <div class='controls'>
                  <a href='/update_consist?schedule_id=${queryData.id}&activity_id=${activity_id}&day=${day}&time=${time}'>수정하기</a> |
                  <a href='/delete_consist?schedule_id=${queryData.id}&activity_id=${activity_id}&day=${day}&time=${time}'>삭제하기</a>
                </div>
              </div>
              `
            }

            return body
          }

          promise.then(function(contents) {
            db.query('SELECT * FROM schedule', function(error, schedules) {
              var schedule_list = template.schedule_list(schedules);
              var review_list = template.review_list(schedules);
              var html = template.HTML(schedule_list, review_list, '', body);
              response.writeHead(200, {'Content-Type': 'text/html'});
              response.end(html);
            })
          })
        })
      })
  } else if (pathname == '/add_consist') {
      var sel_place_id = queryData.sel_place_id;
      db.query(`SELECT * FROM schedule WHERE SCHEDULE_ID = ${queryData.id}`, function(err, schedules) {
        if (err) {
          throw err;
        }
        var body = `<div class='schedule_name'>[${schedules[0].SCHEDULE_COUNTRY}]${schedules[0].SCHEDULE_NAME}</div>
                  <div class='schedule_description'>✈️${schedules[0].SCHEDULE_DESCRIPTION}</div>
                  <hr>`

        var schedule_country = schedules[0].SCHEDULE_COUNTRY;

        db.query(`SELECT * FROM place WHERE PLACE_COUNTRY LIKE ?`, [schedule_country], function(error, places) {
          if (error) {
            throw error;
          }


          if (places[0] == undefined) {
            response.writeHead(302, {'Location': `/schedule?id=${queryData.id}`});
            response.end();
          } else {
            if (sel_place_id === undefined) {
              sel_place_id = places[0].PLACE_ID
            }

            db.query(`SELECT * FROM activity WHERE PLACE_ID=${sel_place_id}`, function(err_act, activities) {
              if (err_act) {
                throw err_act;
              }
              body += `
                  <script>
                    window.onload = function() {
                      document.getElementById('place_select').value = ${sel_place_id}
                    }
                  </script>
                  <div class="activity">
                  <div class="text_section">
                    <form action="/get_activities" method="post">
                      장소
                      <select id='place_select' name='place_id' onchange="this.form.submit()">
                        ${template.placeCombobox(places)}
                      </select>
                      <input type="hidden" name="schedule_id" value=${queryData.id}>
                    </form>

                    <form action="/add_consist_process" method="post">
                      <div class="activity_name">
                        할일
                        <input type="hidden" name="???" value="???">
                        <select name='activity_id'>
                          ${template.activityCombobox(activities)}
                        </select>
                        <input type="hidden" name="schedule_id" value=${queryData.id}>
                        </div>
                        <div class="activity_time">시간 <input type="number" name="day">일차
                        <select name="time">${template.timebox("00:00:00")}</select></div>
                  </div>
                  <div class="submit_button"><input type ="submit" value="추가"></div>
                </form>`

              db.query('SELECT * FROM schedule', function(error, schedules) {
                var schedule_list = template.schedule_list(schedules);
                var review_list = template.review_list(schedules);
                var html = template.HTML(schedule_list, review_list, '', body);
                response.writeHead(200, {'Content-Type': 'text/html'});
                response.end(html);
              })
            })
          }

        })
      })
  } else if (pathname == '/add_consist_process') {
      request.on('data', function(data) {
        body = body + data;
      })
      request.on('end', function() {
        var post = qs.parse(body);
        var schedule_id = post.schedule_id;
        var activity_id = post.activity_id;
        var day = post.day;
        var time = post.time;

        db.query(`INSERT INTO consists VALUES(?,?,?,?)`, [activity_id, schedule_id, time, day], function(err, result) {
          if (err) {
            throw err;
          }
          response.writeHead(302, {Location:`/schedule?id=${schedule_id}`})
          response.end()
        })
      })
  } else if (pathname == '/update_consist') {
      var body = '';
      var schedule_id = queryData.schedule_id;
      var activity_id = queryData.activity_id;
      var day = queryData.day;
      var time = queryData.time;

      db.query(`SELECT * FROM schedule WHERE SCHEDULE_ID = ${schedule_id}`, function(err2,schedule) {
        var schedule_name = schedule[0].SCHEDULE_NAME
        var schedule_description = schedule[0].SCHEDULE_DESCRIPTION
        var schedule_country = schedule[0].SCHEDULE_COUNTRY
        db.query(`SELECT * FROM consists JOIN activity USING (ACTIVITY_ID) JOIN place USING (PLACE_ID)
        WHERE SCHEDULE_ID = ${schedule_id} AND ACTIVITY_ID = ${activity_id} AND CONSISTS_DAY = ${day} AND CONSISTS_TIME = ?`,
                                  [time], function(err,consists) {

          if (err) {
            throw err;
          }
          body += `<div class='schedule_name'>[${schedule_country}]${schedule_name}</div>
                    <div class='schedule_description'>✈️${schedule_description}</div>
                    <hr>
                    `;
          var activity_name = consists[0].ACTIVITY_NAME;
          var activity_description = consists[0].ACTIVITY_DESCRIPTION;
          var activity_image = consists[0].ACTIVITY_IMAGE;
          var place_name = consists[0].PLACE_NAME;

          body += `
            <div class="activity">
            <form action="/update_consist_process" method="post" enctype="multipart/form-data">
              <div class="text_section">
              <div class="activity_time">시간 <input type="number" name="day" value=${day}>일차🕒
                    <select name="time">${template.timebox(time)}</select></div> <br>
              <div class="place_name">${place_name}</div><br>
              <input type="hidden" name="day_before" value=${day}>
              <input type="hidden" name="time_before" value=${time}>
              <input type="hidden" name="activity_id" value=${activity_id}>
              <input type="hidden" name="schedule_id" value=${schedule_id}>
              <div class="activity_name">활동명<br><input type="text" name="activity_name" value=${activity_name}></div> <br>
              <div class="activity_description">활동내용<br><textarea cols="18" rows="5" name="activity_description">${activity_description}</textarea></div></div>`

          if (activity_image != null) {
            body += `<div class="activity_image"><img src=${activity_image}>`
          } else {
            body += `<div class="activity_image">`
          }

          body += `<input type="file" name="activity_image" accept=".png, .jpg, .jpeg"></div>
          <div class="submit_button"><input type ="submit" value="저장"></div></form></div>`

        db.query('SELECT * FROM schedule', function(error, schedules) {
          var schedule_list = template.schedule_list(schedules);
          var review_list = template.review_list(schedules);
          var html = template.HTML(schedule_list, review_list, '', body);
          response.writeHead(200, {'Content-Type': 'text/html'});
          response.end(html);
        })
      })
    })
  } else if (pathname == '/update_consist_process') {
      var body = '';
      var form = new multiparty.Form();

      form.parse(request, function(err, fields, files) {
        if (err) {
          throw err;
        }

        var activity_id = fields.activity_id[0];
        var schedule_id = fields.schedule_id[0];
        var activity_name = fields.activity_name[0];
        var activity_description = fields.activity_description[0];
        var day = fields.day[0];
        var time = fields.time[0];
        var day_before = fields.day_before[0];
        var time_before = fields.time_before[0];

        db.query(`UPDATE activity SET ACTIVITY_NAME=?, ACTIVITY_DESCRIPTION=? WHERE ACTIVITY_ID = ?`,
                      [activity_name, activity_description, activity_id], function(err_act, result) {
          if (err_act) {
            throw err_act;
          }

          db.query(`UPDATE consists SET CONSISTS_DAY=?, CONSISTS_TIME=?
                      WHERE ACTIVITY_ID=? AND SCHEDULE_ID=? AND CONSISTS_DAY=? AND CONSISTS_TIME=?`,
                      [day, time, activity_id, schedule_id, day_before, time_before], function(err_con, result_con) {
            if (err_con) {
              throw err_con;
            }

            if (files.activity_image[0].size == 0) {
              // 사진이 수정되지 않은 경우
              response.writeHead(302, {Location: `/schedule?id=${fields.schedule_id[0]}`});
              response.end();
            } else {
              // 사진이 수정된 경우
              fs.readFile(`${files.activity_image[0].path}`, function (err, data) {
                fs.writeFile(`./images/${activity_id}${files.activity_image[0].path.substring(files.activity_image[0].path.indexOf('.'))}`, data, function(err_write, data) {
                  if (err_write) {
                    throw err_write;
                  }
                  response.writeHead(302, {Location: `/schedule?id=${fields.schedule_id[0]}`});
                  response.end();
                })
              })
            }
          })
        })
      })
  } else if (pathname == '/delete_consist') {
      var schedule_id = queryData.schedule_id;
      var activity_id = queryData.activity_id;
      var day = queryData.day;
      var time = queryData.time;

      db.query('DELETE FROM consists WHERE SCHEDULE_ID = ? AND ACTIVITY_ID = ? AND CONSISTS_DAY = ? AND CONSISTS_TIME = ?',
                  [schedule_id, activity_id, day, time], function(err, result) {
        if (err) {
          throw err;
        }
        response.writeHead(302, {Location : `/update_schedule?id=${schedule_id}`});
        response.end();
      })
  } else if (pathname == '/get_activities') {
      var body = '';

      request.on('data', function(data) {
        body = body + data;
      })
      request.on('end', function() {
        var post = qs.parse(body);

        response.writeHead(302, {Location:`/add_consist?id=${post.schedule_id}&sel_place_id=${post.place_id}`})
        response.end()
      })
  } else if (pathname == '/add_place') {
    // TODO 이름, 나라 입력
  } else if (pathname == '/add_activity') {
    // TODO 드롭박스에서 place 선택하는 방식(select -> optgroup 활용), name, description, image 입력
  } else if (pathname == '/images') {
      fs.readFile(`./images/${queryData.image}`, function(image_error, data){
        // TODO 이미지 없을때 예외처리
        // if (image_error) {
        //   throw image_error;
        // }
        response.writeHead(200, {'Content-Type' : 'image/jpeg'});
        response.end(data);
      });
  } else if (pathname == '/css') {
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

// TODO submenu로 place, activity 추가하는거 만들기

app.listen(3000);
