// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.7.4;

contract ArithmeticOperations {
    //Addition
    function Addition(uint256 n1, uint256 n2) public pure returns (uint256) {
        return (n1 + n2);
    }

    //Subtraction
    function Subtraction(uint256 n1, uint256 n2) public pure returns (uint256) {
        return (n1 - n2);
    }

    //Multiplication
    function Multiplication(uint256 n1, uint256 n2) public pure returns (uint256) {
        return (n1 * n2);
    }

    //Division
    function Division(uint256 n1, uint256 n2) public pure returns (uint256) {
        if (n2 == 0) revert("Division by zero is not allowed");
        return (n1 / n2);
    }

    //Modulus
    function Modulus(uint256 n1, uint256 n2) public pure returns(uint256){
        if (n2 == 0) revert("Division by zero is not allowed");
        return (n1 % n2);
    }    
}
