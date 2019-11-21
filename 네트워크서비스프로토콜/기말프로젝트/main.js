const mysql = require('mysql');
const pw = require('./pw.js');
const fs = require('fs');
const qs = require('querystring');
const url = require('url');
const path = require('path');
const template = require('./lib/template.js');
const async = require('async');
const multiparty = require('multiparty');
const util = require('util');
const express = require('express');

var db = mysql.createConnection({
  host : 'localhost',
  user : 'root',
  password : pw.pw,
  database : 'travel_schedule'
});
db.connect();

/* TODO
  form 꾸미기
  각종 form input 제약설정(사전, 사후) + mysql 오류핸들링?
  해당 schedule의 나라에 place/activity 하나도 없을 경우 오류메시지
  mysql INSERT 성공메시지
  삭제할때 확인 메세지
*/

const app  = express()

// express.static
app.use(express.static('public'))

// 404 ERROR
app.use(function(req, res, next) {
  // TODO 404 처리
  res.status(404).send('<h1> 404 ERROR : PAGE NOT FOUND </h1>');
});

// 첫 페이지
app.get('/', function(request, response) {
  db.query('SELECT * FROM schedule', function(error, schedules) {
    var schedule_list = template.schedule_list(schedules);
    var html = template.HTML(schedule_list,'');
    response.send(html);
  })
})

// 지정한 schedule 열람
app.get('/schedule/:schedule_id', function(request, response) {
  var body = '';
  var schedule_id = path.parse(request.params.schedule_id).base;
  db.query(`SELECT * FROM schedule WHERE SCHEDULE_ID = ${schedule_id}`, function(err_sch,schedule) {
    if (err_sch) {
      throw err_sch;
    }
    db.query(`SELECT * FROM consists JOIN activity USING (ACTIVITY_ID) JOIN place USING (PLACE_ID) WHERE SCHEDULE_ID = ${schedule_id}
                  ORDER BY CONSISTS_DAY, CONSISTS_TIME`, function(err_cons,consists) {
      if(err_cons) {
        throw err_cons;
      }

      // 비동기적 수행
      let promise = new Promise(function(resolve, reject) {
        body += addBody(body, consists)
        resolve(body)
      })

      // schedule에 속하는 consists들을 body에 추가
      function addBody(body, consists) {
        body += template.scheduleInfo(schedule[0])
        body += `<div class='controls'>
                    <a href='${schedule_id}/add_consist'>일정 추가하기</a><br>
                    <a href='${schedule_id}/update_schedule'>일정 수정하기</a><br>
                    <a href='${schedule_id}/delete_schedule'>여행일정 전체 삭제하기</a>
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
            body += `<div class="activity_image"><img src=/${activity_image}></div></div>`
          } else {
            body += `<div class="activity_image"></div></div>`
          }

        }

        return body
      }

      promise.then(function(contents) {
        db.query('SELECT * FROM schedule', function(error, schedules) {
          if (error) {
            throw error;
          }
          var schedule_list = template.schedule_list(schedules);
          var html = template.HTML(schedule_list, body);
          response.send(html);
        })
      })
    })
  })
})

// 새로운 schedule 생성 폼
app.get('/create_schedule', function(request, response) {
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
    var html = template.HTML(schedule_list, body);
    response.send(html);
  })
})

// create schedule post 요청에 대한 응답
app.post('/create_schedule_process', function(request, response) {
  var body = '';

  request.on('data', function(data) {
    body = body + data;
  })

  request.on('end', function() {
    var post = qs.parse(body);

    // TODO insert query 제약사항 확인
    db.query(`INSERT INTO schedule (SCHEDULE_NAME, SCHEDULE_DESCRIPTION, SCHEDULE_COUNTRY)
              VALUES(?,?,?)`, [post.name, post.description, post.country], function(err, result) {
      if (err) {
        throw err;
      }
      response.redirect(`/schedule/${result.insertId}`)
    })
  })
})

// schedule 수정 화면
app.get('/schedule/:schedule_id/update_schedule', function(request, response) {
  var body = '';
  var schedule_id = path.parse(request.params.schedule_id).base;
  db.query(`SELECT * FROM schedule WHERE SCHEDULE_ID = ${schedule_id}`, function(err_sch,schedule) {
    if (err_sch) {
      throw err_sch;
    }
    db.query(`SELECT * FROM consists JOIN activity USING (ACTIVITY_ID) JOIN place USING (PLACE_ID) WHERE SCHEDULE_ID = ${schedule_id}
                  ORDER BY CONSISTS_DAY, CONSISTS_TIME`, function(err_cons,consists) {
      if (err_cons) {
        throw err_cons;
      }
      let promise = new Promise(function(resolve, reject) {
        body += addBody(body, consists)
        resolve(body)
      })

      function addBody(body, consists) {
        body += template.scheduleInfo(schedule[0]);
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
            body += `<div class="activity_image"><img src=/${activity_image}></div>`
          } else {
            body += `<div class="activity_image"></div>`
          }

          body += `
            <div class='controls'>
              <a href='update_consist?activity_id=${activity_id}&day=${day}&time=${time}'>수정하기</a> |
              <a href='delete_consist?activity_id=${activity_id}&day=${day}&time=${time}'>삭제하기</a>
            </div>
          </div>
          `
        }

        return body
      }

      promise.then(function(contents) {
        db.query('SELECT * FROM schedule', function(error, schedules) {
          if (error) {
            throw error;
          }
          var schedule_list = template.schedule_list(schedules);
          var html = template.HTML(schedule_list, body);
          response.send(html);
        })
      })
    })
  })
})

// schedule 삭제 요청처리
app.get('/schedule/:schedule_id/delete_schedule', function(request, response) {
  var schedule_id = path.parse(request.params.schedule_id).base;

  db.query(`DELETE FROM schedule WHERE SCHEDULE_ID=${schedule_id}`, function(err, result) {
    if (err) {
      throw err;
    }
    response.redirect('/');
  })
})

// schedule에 새로운 세부일정 추가하는 화면
app.get('/schedule/:schedule_id/add_consist', function(request, response) {
  var schedule_id = path.parse(request.params.schedule_id).base;
  var sel_place_id = request.query.sel_place_id;
  db.query(`SELECT * FROM schedule WHERE SCHEDULE_ID = ${schedule_id}`, function(err_sch, schedules) {
    if (err_sch) {
      throw err_sch;
    }
    var body = template.scheduleInfo(schedules[0])

    var schedule_country = schedules[0].SCHEDULE_COUNTRY;

    db.query(`SELECT * FROM place WHERE PLACE_COUNTRY LIKE ?`, [schedule_country], function(err_plc, places) {
      if (err_plc) {
        throw err_plc;
      }

      // 해당 스케쥴의 나라와 일치하는 place가 없는경우
      if (places[0] == undefined) {
        response.redirect(`/schedule/${schedule_id}`);
      }

      // 일치하는 place가 하나라도 있는 경우
      else {
        // place query가 없는 경우(select에서 선택 X)
        if (sel_place_id === undefined) {
          // 기본값으로 첫번째 option 선택
          sel_place_id = places[0].PLACE_ID
        }

        db.query(`SELECT * FROM activity WHERE PLACE_ID=${sel_place_id}`, function(err_act, activities) {
          if (err_act) {
            throw err_act;
          }
          // 첫번째 폼 : select 중 하나 선택시 place값을 갱신하고 refresh
          // 두번째 폼 : consist를 데이터베이스에 추가
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
                  <input type="hidden" name="schedule_id" value=${schedule_id}>
                </form>

                <form action="/add_consist_process" method="post">
                  <div class="activity_name">
                    할일
                    <input type="hidden" name="???" value="???">
                    <select name='activity_id'>
                      ${template.activityCombobox(activities)}
                    </select>
                    <input type="hidden" name="schedule_id" value=${schedule_id}>
                    </div>
                    <div class="activity_time">시간 <input type="number" name="day">일차
                    <select name="time">${template.timebox("00:00:00")}</select></div>
              </div>
              <div class="submit_button"><input type ="submit" value="추가"></div>
            </form>`

          db.query('SELECT * FROM schedule', function(error, schedules) {
            var schedule_list = template.schedule_list(schedules);
            var html = template.HTML(schedule_list, body);
            response.send(html);
          })
        })
      }

    })
  })
})

// place에 해당하는 activity를 반환하는 페이지로 리다이렉트
app.post('/get_activities', function(request, response) {
  var body = '';

  request.on('data', function(data) {
    body = body + data;
  })
  request.on('end', function() {
    var post = qs.parse(body);

    response.redirect(`/schedule/${post.schedule_id}/add_consist?sel_place_id=${post.place_id}`);
  })
})

// schedule에 새로운 consists 추가(db insert)
app.post('/add_consist_process', function(request, response) {
  var body = '';
  request.on('data', function(data) {
    body = body + data;
  })
  request.on('end', function() {
    var post = qs.parse(body);
    var schedule_id = post.schedule_id;
    var activity_id = post.activity_id;
    var day = post.day;
    var time = post.time;

    db.query(`INSERT INTO consists VALUES(?,?,?,?)`,
                  [activity_id, schedule_id, time, day], function(err_cons, result) {
      if (err_cons) {
        throw err_cons;
      }
      response.redirect(`/schedule/${schedule_id}`)
    })
  })
})

// consist 삭제 처리
app.get('/schedule/:schedule_id/delete_consist', function(request, response) {
    var schedule_id = path.parse(request.params.schedule_id).base;
    var activity_id = request.query.activity_id;
    var day = request.query.day;
    var time = request.query.time;

    db.query('DELETE FROM consists WHERE SCHEDULE_ID = ? AND ACTIVITY_ID = ? AND CONSISTS_DAY = ? AND CONSISTS_TIME = ?',
                [schedule_id, activity_id, day, time], function(err, result) {
      if (err) {
        throw err;
      }
      response.redirect(`/schedule/${schedule_id}/update_schedule`);
    })
})

// consist 수정 폼
app.get('/schedule/:schedule_id/update_consist', function(request, response) {
  var body = '';
  var schedule_id = path.parse(request.params.schedule_id).base;
  var activity_id = request.query.activity_id;
  var day = request.query.day;
  var time = request.query.time;

  db.query(`SELECT * FROM schedule WHERE SCHEDULE_ID = ${schedule_id}`, function(err_sch,schedule) {
    if (err_sch) {
      throw err_sch;
    }
    db.query(`SELECT * FROM consists JOIN activity USING (ACTIVITY_ID) JOIN place USING (PLACE_ID)
    WHERE SCHEDULE_ID = ${schedule_id} AND ACTIVITY_ID = ${activity_id} AND CONSISTS_DAY = ${day} AND CONSISTS_TIME = ?`,
                              [time], function(err_act,consists) {
      if (err_act) {
        throw err_act;
      }

      body += template.scheduleInfo(schedule[0]);
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
          <div class="activity_name">활동명<br>
          <input type="text" name="activity_name" value="${activity_name}"></div> <br>
          <div class="activity_description">활동내용<br><textarea cols="18" rows="5" name="activity_description">${activity_description}</textarea></div></div>`

      if (activity_image != null) {
        body += `<div class="activity_image"><img src=/${activity_image}>`
      } else {
        body += `<div class="activity_image">`
      }

      body += `<input type="file" name="activity_image" accept=".png, .jpg, .jpeg"></div>
      <div class="submit_button"><input type ="submit" value="저장"></div></form></div>`

    db.query('SELECT * FROM schedule', function(error, schedules) {
      var schedule_list = template.schedule_list(schedules);
      var html = template.HTML(schedule_list, body);
      response.send(html);
    })
  })
})
})

// consist 수정 (db update)
app.post('/update_consist_process', function(request, response) {
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
          // 사진이 수정되지 않은 경우 바로 리다이렉트
          response.redirect(`/schedule/${fields.schedule_id[0]}/update_schedule`);
        } else {
          // 사진이 수정된 경우, 새로 저장한 후 리다이렉트
          fs.readFile(`${files.activity_image[0].path}`, function (err, data) {
            fs.writeFile(`./images/${activity_id}${files.activity_image[0].path.substring(files.activity_image[0].path.indexOf('.'))}`, data, function(err_write, data) {
              if (err_write) {
                throw err_write;
              }
              response.redirect(`/schedule/${fields.schedule_id[0]}/update_schedule`);
            })
          })
        }
      })
    })
  })
})

// place 추가 폼
app.get('/add_place', function(request, response) {
  db.query('SELECT * FROM schedule', function(error, schedules) {
    var body = `
    <form action="/add_place_process" method="post">
      <p>나라이름</p>
      <p><input type="text" name="place_country" placeholder="Place Country"></p>
      <p>여행지 이름</p>
      <p><input type="text" name="place_name" placeholder="Place Name"></p>
      <p><input type="submit" value="추가"></p>
    </form>
    `
    var schedule_list = template.schedule_list(schedules);
    var html = template.HTML(schedule_list, body);
    response.send(html);
  })
})

// place 추가 (db insert)
app.post('/add_place_process', function(request, response) {
  var body = '';

  request.on('data', function(data) {
    body += data;
  })

  request.on('end', function() {
    var post = qs.parse(body);
    var place_name = post.place_name;
    var place_country = post.place_country;

    db.query(`INSERT INTO place (PLACE_NAME, PLACE_COUNTRY) VALUES(?,?)`, [place_name, place_country], function(err_plc, result) {
      if (err_plc) {
        throw err_plc;
      }
      response.redirect('/')
    })
  })
})

// activity 추가 폼
app.get('/add_activity', function(request, response) {
  db.query('SELECT * FROM place ORDER BY PLACE_COUNTRY', function(err_plc, places) {
    if (err_plc) {
      throw err_plc;
    }
    db.query('SELECT * FROM schedule', function(err_sch, schedules) {
      if (err_sch) {
        throw err_sch;
      }
      var body = `
      <form action="/add_activity_process" method="post" enctype="multipart/form-data">
        <p>활동 나라</p>
        <p><select name="activity_place">${template.placeComboboxSub(places)}</select>
        <p>활동 이름</p>
        <p><input type="text" name="activity_name" placeholder="Activity Name"></p>
        <p>활동 설명</p>
        <p><textarea name="activity_description" placeholder="Activity Description"></textarea></p>
        <p>활동 사진</p>
        <input type="file" name="activity_image" accept=".png, .jpg, .jpeg">
        <p><input type="submit" value="추가"></p>
      </form>
      `
      var schedule_list = template.schedule_list(schedules);
      var html = template.HTML(schedule_list, body);
      response.send(html);
    })
  })
})

// place 추가 (db insert)
app.post('/add_activity_process', function(request, response) {
  var body = ''
  var form = new multiparty.Form()
  form.parse(request, function(err, fields, files) {
    if (err) {
      throw err;
    }

    var activity_name = fields.activity_name[0];
    var activity_description = fields.activity_description[0];
    var place_id = fields.activity_place[0];

    db.query(`INSERT INTO activity (PLACE_ID, ACTIVITY_NAME, ACTIVITY_DESCRIPTION) VALUES(?,?,?)`,
                [place_id, activity_name, activity_description], function(err_act, result) {
      if (err_act) {
        throw err_act;
      }

      fs.readFile(`${files.activity_image[0].path}`, function (err_read, data) {
        if (err_read) {
          throw err_read;
        }
        var filename = `./images/${result.insertId}${files.activity_image[0].path.substring(files.activity_image[0].path.indexOf('.'))}`
        fs.writeFile(filename, data, function(err_write, data) {
          if (err_write) {
            throw err_write;
          }
          db.query(`UPDATE activity SET ACTIVITY_IMAGE = ? WHERE ACTIVITY_ID = ${result.insertId}`, [filename], function(err, result_img) {
            response.redirect(`/`);
          })
        })
      })
    })
  })
})

// place 삭제 폼
app.get('/delete_place', function(request, response) {
  db.query('SELECT * FROM place', function(err_plc, places) {
    if (err_plc) {
      throw err_plc;
    }
    db.query('SELECT * FROM schedule', function(error, schedules) {
      if (error) {
        throw error;
      }

      var body = `
      <form action="/delete_place_process" method="post">
        <p>여행지 이름</p>
        <p><select name="place_id">${template.placeComboboxSub(places)}</select>
        <div class="warning">주의 : 해당 여행지에서 진행되는 활동 데이터가 모두 지워집니다!</div>
        <input type="submit" value="삭제">
      </form>
      `
      var schedule_list = template.schedule_list(schedules);
      var html = template.HTML(schedule_list, body);
      response.send(html);
    })
  })
})

// place 삭제 (db delete)
app.post('/delete_place_process', function(request, response) {
  var body = '';
  request.on('data', function(data) {
    body += data;
  })
  request.on('end', function() {
    var post = qs.parse(body);
    db.query(`DELETE FROM place WHERE PLACE_ID=${post.place_id}`, function(err, result) {
      if (err) {
        throw err;
      }
      response.redirect('/');
    })
  })
})

// activity 삭제 폼
app.get('/delete_activity', function(request, response) {
  var body = ''
  var sel_place_id = request.query.sel_place_id;
  db.query(`SELECT * FROM place ORDER BY PLACE_COUNTRY`, function(err_plc, places) {
    if (err_plc) {
      throw err_plc;
    }

    if (sel_place_id === undefined) {
      sel_place_id = places[0].PLACE_ID
    }
    db.query('SELECT * FROM activity WHERE PLACE_ID = ?', [sel_place_id], function(err_act, activities) {
      if (err_act) {
        throw err_act;
      }

      // 첫번째 폼 : select 중 하나 선택시 place값을 갱신하고 refresh
      // 두번째 폼 : 삭제진행
      body += `
        <form action="/get_activities_place" method="post">
          장소
          <select id='place_select' name='place_id' onchange="this.form.submit()">
            ${template.placeComboboxSub(places, sel_place_id)}
          </select>
        </form>

        <form action="/delete_activity_process" method="post">
            <select name='activity_id'>
              ${template.activityCombobox(activities)}
            </select>
            <div class="submit_button"><input type ="submit" value="삭제">
        </form>
        `

      db.query('SELECT * FROM schedule', function(error, schedules) {
        if (error) {
          throw error;
        }
        var schedule_list = template.schedule_list(schedules);
        var html = template.HTML(schedule_list, body);
        response.send(html);
      })
    })
  })
})

app.post('/get_activities_place', function (request, response) {
  var body = '';

  request.on('data', function(data) {
    body = body + data;
  })
  request.on('end', function() {
    var post = qs.parse(body);

    response.redirect(`/delete_activity?sel_place_id=${post.place_id}`);
  })
})

// activity 삭제 (db delete)
app.post('/delete_activity_process', function(request, response) {
  var body = '';
  request.on('data', function(data) {
    body += data;
  })
  request.on('end', function() {
    var post = qs.parse(body);
    db.query(`SELECT * FROM activity WHERE ACTIVITY_ID=${post.activity_id}`, function(err_act,activity) {
      db.query(`DELETE FROM activity WHERE ACTIVITY_ID=${post.activity_id}`, function(err, result) {
        if (err) {
          throw err;
        }
        fs.unlink(activity[0].ACTIVITY_IMAGE, function(err_img) {
          if (err_img) {
            throw err_img;
          }
          response.redirect('/');
        })
      })
    })
  })
})

// 서버 실행
app.listen(3000, () => console.log('Server Running on port number 3000'))
