// SPDX-License-Identifier: INLICENSED
pragma solidity ^0.7.4;

contract FirstProgram {
    string message = "Message: Welcome to Solidity Programming!";
    string name = "Name: Santosh Kandalkar";
    string rollNo = "Roll No: 74";
    string branch = "Branch: Computer Science and Engineering";

    function GetData()
        public view returns (string memory, string memory, string memory, string memory){
        return (message, name, rollNo, branch);
    }
}