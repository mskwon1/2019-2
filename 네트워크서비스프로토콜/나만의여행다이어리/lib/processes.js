const express = require('express')
const pw = require('./pw.js')
const mysql = require('mysql')
const bodyParser = require('body-parser')
const multiparty = require('multiparty')
const fs = require('fs')

var app = express()
var router = express.Router()

var db = mysql.createConnection({
  host : 'localhost',ㄱ
  user : 'root',
  password : pw.pw,
  database : 'travel_schedule'
});
db.connect();

// create schedule post 요청에 대한 응답
router.post('/create_schedule_process', function(request, response) {
  var post = request.body;

  // TODO insert query 제약사항 확인
  db.query(`INSERT INTO schedule (SCHEDULE_NAME, SCHEDULE_DESCRIPTION, SCHEDULE_COUNTRY)
            VALUES(?,?,?)`, [post.name, post.description, post.country], function(err, result) {
    if (err) {
      throw err;
    }
    response.redirect(`/schedule/${result.insertId}`)
  })
})

// place에 해당하는 activity를 반환하는 페이지로 리다이렉트
router.post('/get_activities', function(request, response) {
  var post = request.body;
  response.redirect(`/schedule/${post.schedule_id}/add_consist?sel_place_id=${post.place_id}`);
})

// schedule에 새로운 consists 추가(db insert)
router.post('/add_consist_process', function(request, response) {
  var post = request.body;
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

// consist 수정 (db update)
router.post('/update_consist_process', function(request, response) {
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
          db.query(`SELECT * FROM activity WHERE ACTIVITY_ID = ${activity_id}`, function(err_act2, activity) {
            fs.unlink('./public/' + activity[0].ACTIVITY_IMAGE, function(err_img) {
              if (err_img) {
                throw err_img;
              }
              fs.readFile(`${files.activity_image[0].path}`, function (err, data) {
                if (err) {
                  throw err;
                }
                var filename = `./public/images/${activity_id}${files.activity_image[0].path.substring(files.activity_image[0].path.indexOf('.'))}`
                var savename = `./images/${activity_id}${files.activity_image[0].path.substring(files.activity_image[0].path.indexOf('.'))}`

                fs.writeFile(filename, data, function(err_write, data) {
                  if (err_write) {
                    throw err_write;
                  }
                  db.query(`UPDATE activity SET ACTIVITY_IMAGE = ? WHERE ACTIVITY_ID = ${activity_id}`,
                            [savename], function(err, result_img) {
                    if (err) {
                      throw err;
                    }
                    response.redirect(`/schedule/${schedule_id}/update_schedule`);
                  })
                })
              })
            })
          })
        }
      })
    })
  })
})

// place 추가 (db insert)
router.post('/add_place_process', function(request, response) {
  var post = request.body;
  var place_name = post.place_name;
  var place_country = post.place_country;

  db.query(`INSERT INTO place (PLACE_NAME, PLACE_COUNTRY) VALUES(?,?)`, [place_name, place_country], function(err_plc, result) {
    if (err_plc) {
      throw err_plc;
    }
    response.redirect('/')
  })
})

// activity 추가 (db insert)
router.post('/add_activity_process', function(request, response) {
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
        var filename = `./public/images/${result.insertId}${files.activity_image[0].path.substring(files.activity_image[0].path.indexOf('.'))}`
        var savename = `./images/${result.insertId}${files.activity_image[0].path.substring(files.activity_image[0].path.indexOf('.'))}`
        fs.writeFile(filename, data, function(err_write, data) {
          if (err_write) {
            throw err_write;
          }
          db.query(`UPDATE activity SET ACTIVITY_IMAGE = ? WHERE ACTIVITY_ID = ${result.insertId}`,
                    [savename], function(err, result_img) {
            response.redirect(`/`);
          })
        })
      })
    })
  })
})

// place 삭제 (db delete)
router.post('/delete_place_process', function(request, response) {
  var post = request.body;
  db.query(`DELETE FROM place WHERE PLACE_ID=${post.place_id}`, function(err, result) {
    if (err) {
      throw err;
    }
    response.redirect('/');
  })
})

// redirection for select value change
router.post('/get_activities_place', function (request, response) {
  var post = request.body;
  response.redirect(`/delete_activity?sel_place_id=${post.place_id}`);
})

// activity 삭제 (db delete)
router.post('/delete_activity_process', function(request, response) {
  var post = request.body;
  db.query(`SELECT * FROM activity WHERE ACTIVITY_ID=${post.activity_id}`, function(err_act,activity) {
    db.query(`DELETE FROM activity WHERE ACTIVITY_ID=${post.activity_id}`, function(err, result) {
      if (err) {
        throw err;
      }
      fs.unlink('./public/' + activity[0].ACTIVITY_IMAGE, function(err_img) {
        if (err_img) {
          throw err_img;
        }
        response.redirect('/');
      })
    })
  })
})

module.exports = router
