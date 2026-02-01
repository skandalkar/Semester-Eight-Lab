// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.7.4;

contract Test {
    function callKeccak256() public pure returns (bytes32 result) {
        return keccak256("ABC");
    }
}
