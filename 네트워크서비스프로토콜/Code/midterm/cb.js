function findUser(id) {
    const user = {
        id : id,
        name : "User" + id,
        email : id + "@test.com"
    };
    return user;
}

function findUserAndCallBack(id, userFunc) {
  return userFunc(id);
}

const user = findUserAndCallBack(1, findUser);
console.log("user:", user);
