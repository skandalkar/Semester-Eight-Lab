// SPDX-License-Identifier: UNLICENSED

pragma solidity ^0.7.4;

contract CryptoFunctions {

    //keccak256 Hash Function
    function callKeccak256() public pure returns (bytes32) {
        return keccak256(abi.encodePacked("ABC"));
    }

    //sha256 Hash Function
    function callSha256() public pure returns (bytes32) {
        return sha256(abi.encodePacked("ABC"));
    }

    //ripemd160 Hash Function
    function callRipemd160() public pure returns (bytes20) {
        return ripemd160(abi.encodePacked("ABC"));
    }

    //ecrecover Function
    function callEcrecover(
        bytes32 hash,
        uint8 v,
        bytes32 r,
        bytes32 s
    ) public pure returns (address) {
        return ecrecover(hash, v, r, s);
    }
}