
Object.defineProperty(exports, "__esModule", { value: true });

const {
  PrismaClientKnownRequestError,
  PrismaClientUnknownRequestError,
  PrismaClientRustPanicError,
  PrismaClientInitializationError,
  PrismaClientValidationError,
  NotFoundError,
  getPrismaClient,
  sqltag,
  empty,
  join,
  raw,
  Decimal,
  Debug,
  objectEnumValues,
  makeStrictEnum,
  Extensions,
  warnOnce,
  defineDmmfProperty,
  Public,
} = require('./runtime/edge')


const Prisma = {}

exports.Prisma = Prisma
exports.$Enums = {}

/**
 * Prisma Client JS version: 5.4.2
 * Query Engine version: ac9d7041ed77bcc8a8dbd2ab6616b39013829574
 */
Prisma.prismaVersion = {
  client: "5.4.2",
  engine: "ac9d7041ed77bcc8a8dbd2ab6616b39013829574"
}

Prisma.PrismaClientKnownRequestError = PrismaClientKnownRequestError;
Prisma.PrismaClientUnknownRequestError = PrismaClientUnknownRequestError
Prisma.PrismaClientRustPanicError = PrismaClientRustPanicError
Prisma.PrismaClientInitializationError = PrismaClientInitializationError
Prisma.PrismaClientValidationError = PrismaClientValidationError
Prisma.NotFoundError = NotFoundError
Prisma.Decimal = Decimal

/**
 * Re-export of sql-template-tag
 */
Prisma.sql = sqltag
Prisma.empty = empty
Prisma.join = join
Prisma.raw = raw
Prisma.validator = Public.validator

/**
* Extensions
*/
Prisma.getExtensionContext = Extensions.getExtensionContext
Prisma.defineExtension = Extensions.defineExtension

/**
 * Shorthand utilities for JSON filtering
 */
Prisma.DbNull = objectEnumValues.instances.DbNull
Prisma.JsonNull = objectEnumValues.instances.JsonNull
Prisma.AnyNull = objectEnumValues.instances.AnyNull

Prisma.NullTypes = {
  DbNull: objectEnumValues.classes.DbNull,
  JsonNull: objectEnumValues.classes.JsonNull,
  AnyNull: objectEnumValues.classes.AnyNull
}



/**
 * Enums
 */
exports.Prisma.TransactionIsolationLevel = makeStrictEnum({
  ReadUncommitted: 'ReadUncommitted',
  ReadCommitted: 'ReadCommitted',
  RepeatableRead: 'RepeatableRead',
  Serializable: 'Serializable'
});

exports.Prisma.DATABASECHANGELOGLOCKScalarFieldEnum = {
  ID: 'ID',
  LOCKED: 'LOCKED',
  LOCKGRANTED: 'LOCKGRANTED',
  LOCKEDBY: 'LOCKEDBY'
};

exports.Prisma.SESSIONSScalarFieldEnum = {
  UID: 'UID',
  createdAt: 'createdAt',
  USER: 'USER',
  rec_garage: 'rec_garage',
  rec_avail_1: 'rec_avail_1',
  rec_travel_time_1: 'rec_travel_time_1',
  rec_avail_2: 'rec_avail_2',
  rec_travel_time_2: 'rec_travel_time_2',
  rec_avail_3: 'rec_avail_3',
  rec_travel_time_3: 'rec_travel_time_3'
};

exports.Prisma.USERSScalarFieldEnum = {
  UID: 'UID',
  createdAt: 'createdAt',
  name: 'name',
  admin: 'admin',
  username: 'username',
  password: 'password',
  parking_pass_type: 'parking_pass_type',
  address: 'address',
  bio: 'bio'
};

exports.Prisma.SortOrder = {
  asc: 'asc',
  desc: 'desc'
};

exports.Prisma.NullsOrder = {
  first: 'first',
  last: 'last'
};


exports.Prisma.ModelName = {
  DATABASECHANGELOGLOCK: 'DATABASECHANGELOGLOCK',
  SESSIONS: 'SESSIONS',
  USERS: 'USERS'
};
/**
 * Create the Client
 */
const config = {
  "generator": {
    "name": "client",
    "provider": {
      "fromEnvVar": null,
      "value": "prisma-client-js"
    },
    "output": {
      "value": "C:\\Users\\Atul\\Documents\\Parking-Predictor-2025\\persistence\\prisma\\generated\\client",
      "fromEnvVar": null
    },
    "config": {
      "engineType": "library"
    },
    "binaryTargets": [
      {
        "fromEnvVar": null,
        "value": "windows",
        "native": true
      }
    ],
    "previewFeatures": [],
    "isCustomOutput": true
  },
  "relativeEnvPaths": {
    "rootEnvPath": "../../../.env",
    "schemaEnvPath": "../../../.env"
  },
  "relativePath": "../..",
  "clientVersion": "5.4.2",
  "engineVersion": "ac9d7041ed77bcc8a8dbd2ab6616b39013829574",
  "datasourceNames": [
    "db"
  ],
  "activeProvider": "mysql",
  "postinstall": true,
  "inlineDatasources": {
    "db": {
      "url": {
        "fromEnvVar": "DATABASE_URL",
        "value": null
      }
    }
  },
  "inlineSchema": "Z2VuZXJhdG9yIGNsaWVudCB7DQogIHByb3ZpZGVyID0gInByaXNtYS1jbGllbnQtanMiDQogIG91dHB1dCAgID0gIi4vZ2VuZXJhdGVkL2NsaWVudCINCn0NCg0KZGF0YXNvdXJjZSBkYiB7DQogIHByb3ZpZGVyID0gIm15c3FsIg0KICB1cmwgICAgICA9IGVudigiREFUQUJBU0VfVVJMIikNCn0NCg0KLy8vIFRoZSB1bmRlcmx5aW5nIHRhYmxlIGRvZXMgbm90IGNvbnRhaW4gYSB2YWxpZCB1bmlxdWUgaWRlbnRpZmllciBhbmQgY2FuIHRoZXJlZm9yZSBjdXJyZW50bHkgbm90IGJlIGhhbmRsZWQgYnkgUHJpc21hIENsaWVudC4NCm1vZGVsIERBVEFCQVNFQ0hBTkdFTE9HIHsNCiAgSUQgICAgICAgICAgICBTdHJpbmcgICBAZGIuVmFyQ2hhcigyNTUpDQogIEFVVEhPUiAgICAgICAgU3RyaW5nICAgQGRiLlZhckNoYXIoMjU1KQ0KICBGSUxFTkFNRSAgICAgIFN0cmluZyAgIEBkYi5WYXJDaGFyKDI1NSkNCiAgREFURUVYRUNVVEVEICBEYXRlVGltZSBAZGIuRGF0ZVRpbWUoMCkNCiAgT1JERVJFWEVDVVRFRCBJbnQNCiAgRVhFQ1RZUEUgICAgICBTdHJpbmcgICBAZGIuVmFyQ2hhcigxMCkNCiAgTUQ1U1VNICAgICAgICBTdHJpbmc/ICBAZGIuVmFyQ2hhcigzNSkNCiAgREVTQ1JJUFRJT04gICBTdHJpbmc/ICBAZGIuVmFyQ2hhcigyNTUpDQogIENPTU1FTlRTICAgICAgU3RyaW5nPyAgQGRiLlZhckNoYXIoMjU1KQ0KICBUQUcgICAgICAgICAgIFN0cmluZz8gIEBkYi5WYXJDaGFyKDI1NSkNCiAgTElRVUlCQVNFICAgICBTdHJpbmc/ICBAZGIuVmFyQ2hhcigyMCkNCiAgQ09OVEVYVFMgICAgICBTdHJpbmc/ICBAZGIuVmFyQ2hhcigyNTUpDQogIExBQkVMUyAgICAgICAgU3RyaW5nPyAgQGRiLlZhckNoYXIoMjU1KQ0KICBERVBMT1lNRU5UX0lEIFN0cmluZz8gIEBkYi5WYXJDaGFyKDEwKQ0KDQogIEBAaWdub3JlDQp9DQoNCm1vZGVsIERBVEFCQVNFQ0hBTkdFTE9HTE9DSyB7DQogIElEICAgICAgICAgIEludCAgICAgICBAaWQNCiAgTE9DS0VEICAgICAgQm9vbGVhbiAgIEBkYi5CaXQoMSkNCiAgTE9DS0dSQU5URUQgRGF0ZVRpbWU/IEBkYi5EYXRlVGltZSgwKQ0KICBMT0NLRURCWSAgICBTdHJpbmc/ICAgQGRiLlZhckNoYXIoMjU1KQ0KfQ0KDQptb2RlbCBTRVNTSU9OUyB7DQogIFVJRCAgICAgICAgICAgICAgIEludCAgICAgICBAaWQgQGRlZmF1bHQoYXV0b2luY3JlbWVudCgpKQ0KICBjcmVhdGVkQXQgICAgICAgICBEYXRlVGltZT8gQGRiLlRpbWVzdGFtcCgwKQ0KICBVU0VSICAgICAgICAgICAgICBJbnQ/DQogIHJlY19nYXJhZ2UgICAgICAgIFN0cmluZz8gICBAZGIuVmFyQ2hhcigyNTUpDQogIHJlY19hdmFpbF8xICAgICAgIEludD8NCiAgcmVjX3RyYXZlbF90aW1lXzEgSW50Pw0KICByZWNfYXZhaWxfMiAgICAgICBJbnQ/DQogIHJlY190cmF2ZWxfdGltZV8yIEludD8NCiAgcmVjX2F2YWlsXzMgICAgICAgSW50Pw0KICByZWNfdHJhdmVsX3RpbWVfMyBJbnQ/DQogIFVTRVJTICAgICAgICAgICAgIFVTRVJTPyAgICBAcmVsYXRpb24oZmllbGRzOiBbVVNFUl0sIHJlZmVyZW5jZXM6IFtVSURdLCBvbkRlbGV0ZTogTm9BY3Rpb24sIG9uVXBkYXRlOiBOb0FjdGlvbiwgbWFwOiAiZmtfVVNFUiIpDQoNCiAgQEBpbmRleChbVVNFUl0sIG1hcDogImZrX1VTRVIiKQ0KfQ0KDQptb2RlbCBVU0VSUyB7DQogIFVJRCAgICAgICAgICAgICAgIEludCAgICAgICAgQGlkIEBkZWZhdWx0KGF1dG9pbmNyZW1lbnQoKSkNCiAgY3JlYXRlZEF0ICAgICAgICAgRGF0ZVRpbWU/ICBAZGIuVGltZXN0YW1wKDApDQogIG5hbWUgICAgICAgICAgICAgIFN0cmluZz8gICAgQGRiLlZhckNoYXIoMjU1KQ0KICBhZG1pbiAgICAgICAgICAgICBCb29sZWFuPyAgIEBkYi5CaXQoMSkNCiAgdXNlcm5hbWUgICAgICAgICAgU3RyaW5nPyAgICBAZGIuVmFyQ2hhcigyNTUpDQogIHBhc3N3b3JkICAgICAgICAgIFN0cmluZz8gICAgQGRiLlZhckNoYXIoMjU1KQ0KICBwYXJraW5nX3Bhc3NfdHlwZSBTdHJpbmc/ICAgIEBkYi5WYXJDaGFyKDI1NSkNCiAgYWRkcmVzcyAgICAgICAgICAgU3RyaW5nPyAgICBAZGIuVmFyQ2hhcigyNTUpDQogIGJpbyAgICAgICAgICAgICAgIFN0cmluZz8gICAgQGRiLlRleHQNCiAgU0VTU0lPTlMgICAgICAgICAgU0VTU0lPTlNbXQ0KfQ0K",
  "inlineSchemaHash": "cd9bca4e21777ac500beaf43ab93f9906ad3bac8d6db5a3e56ca0784cb8d16c2",
  "noEngine": false
}
config.dirname = '/'

config.runtimeDataModel = JSON.parse("{\"models\":{\"DATABASECHANGELOGLOCK\":{\"dbName\":null,\"fields\":[{\"name\":\"ID\",\"kind\":\"scalar\",\"isList\":false,\"isRequired\":true,\"isUnique\":false,\"isId\":true,\"isReadOnly\":false,\"hasDefaultValue\":false,\"type\":\"Int\",\"isGenerated\":false,\"isUpdatedAt\":false},{\"name\":\"LOCKED\",\"kind\":\"scalar\",\"isList\":false,\"isRequired\":true,\"isUnique\":false,\"isId\":false,\"isReadOnly\":false,\"hasDefaultValue\":false,\"type\":\"Boolean\",\"isGenerated\":false,\"isUpdatedAt\":false},{\"name\":\"LOCKGRANTED\",\"kind\":\"scalar\",\"isList\":false,\"isRequired\":false,\"isUnique\":false,\"isId\":false,\"isReadOnly\":false,\"hasDefaultValue\":false,\"type\":\"DateTime\",\"isGenerated\":false,\"isUpdatedAt\":false},{\"name\":\"LOCKEDBY\",\"kind\":\"scalar\",\"isList\":false,\"isRequired\":false,\"isUnique\":false,\"isId\":false,\"isReadOnly\":false,\"hasDefaultValue\":false,\"type\":\"String\",\"isGenerated\":false,\"isUpdatedAt\":false}],\"primaryKey\":null,\"uniqueFields\":[],\"uniqueIndexes\":[],\"isGenerated\":false},\"SESSIONS\":{\"dbName\":null,\"fields\":[{\"name\":\"UID\",\"kind\":\"scalar\",\"isList\":false,\"isRequired\":true,\"isUnique\":false,\"isId\":true,\"isReadOnly\":false,\"hasDefaultValue\":true,\"type\":\"Int\",\"default\":{\"name\":\"autoincrement\",\"args\":[]},\"isGenerated\":false,\"isUpdatedAt\":false},{\"name\":\"createdAt\",\"kind\":\"scalar\",\"isList\":false,\"isRequired\":false,\"isUnique\":false,\"isId\":false,\"isReadOnly\":false,\"hasDefaultValue\":false,\"type\":\"DateTime\",\"isGenerated\":false,\"isUpdatedAt\":false},{\"name\":\"USER\",\"kind\":\"scalar\",\"isList\":false,\"isRequired\":false,\"isUnique\":false,\"isId\":false,\"isReadOnly\":true,\"hasDefaultValue\":false,\"type\":\"Int\",\"isGenerated\":false,\"isUpdatedAt\":false},{\"name\":\"rec_garage\",\"kind\":\"scalar\",\"isList\":false,\"isRequired\":false,\"isUnique\":false,\"isId\":false,\"isReadOnly\":false,\"hasDefaultValue\":false,\"type\":\"String\",\"isGenerated\":false,\"isUpdatedAt\":false},{\"name\":\"rec_avail_1\",\"kind\":\"scalar\",\"isList\":false,\"isRequired\":false,\"isUnique\":false,\"isId\":false,\"isReadOnly\":false,\"hasDefaultValue\":false,\"type\":\"Int\",\"isGenerated\":false,\"isUpdatedAt\":false},{\"name\":\"rec_travel_time_1\",\"kind\":\"scalar\",\"isList\":false,\"isRequired\":false,\"isUnique\":false,\"isId\":false,\"isReadOnly\":false,\"hasDefaultValue\":false,\"type\":\"Int\",\"isGenerated\":false,\"isUpdatedAt\":false},{\"name\":\"rec_avail_2\",\"kind\":\"scalar\",\"isList\":false,\"isRequired\":false,\"isUnique\":false,\"isId\":false,\"isReadOnly\":false,\"hasDefaultValue\":false,\"type\":\"Int\",\"isGenerated\":false,\"isUpdatedAt\":false},{\"name\":\"rec_travel_time_2\",\"kind\":\"scalar\",\"isList\":false,\"isRequired\":false,\"isUnique\":false,\"isId\":false,\"isReadOnly\":false,\"hasDefaultValue\":false,\"type\":\"Int\",\"isGenerated\":false,\"isUpdatedAt\":false},{\"name\":\"rec_avail_3\",\"kind\":\"scalar\",\"isList\":false,\"isRequired\":false,\"isUnique\":false,\"isId\":false,\"isReadOnly\":false,\"hasDefaultValue\":false,\"type\":\"Int\",\"isGenerated\":false,\"isUpdatedAt\":false},{\"name\":\"rec_travel_time_3\",\"kind\":\"scalar\",\"isList\":false,\"isRequired\":false,\"isUnique\":false,\"isId\":false,\"isReadOnly\":false,\"hasDefaultValue\":false,\"type\":\"Int\",\"isGenerated\":false,\"isUpdatedAt\":false},{\"name\":\"USERS\",\"kind\":\"object\",\"isList\":false,\"isRequired\":false,\"isUnique\":false,\"isId\":false,\"isReadOnly\":false,\"hasDefaultValue\":false,\"type\":\"USERS\",\"relationName\":\"SESSIONSToUSERS\",\"relationFromFields\":[\"USER\"],\"relationToFields\":[\"UID\"],\"relationOnDelete\":\"NoAction\",\"isGenerated\":false,\"isUpdatedAt\":false}],\"primaryKey\":null,\"uniqueFields\":[],\"uniqueIndexes\":[],\"isGenerated\":false},\"USERS\":{\"dbName\":null,\"fields\":[{\"name\":\"UID\",\"kind\":\"scalar\",\"isList\":false,\"isRequired\":true,\"isUnique\":false,\"isId\":true,\"isReadOnly\":false,\"hasDefaultValue\":true,\"type\":\"Int\",\"default\":{\"name\":\"autoincrement\",\"args\":[]},\"isGenerated\":false,\"isUpdatedAt\":false},{\"name\":\"createdAt\",\"kind\":\"scalar\",\"isList\":false,\"isRequired\":false,\"isUnique\":false,\"isId\":false,\"isReadOnly\":false,\"hasDefaultValue\":false,\"type\":\"DateTime\",\"isGenerated\":false,\"isUpdatedAt\":false},{\"name\":\"name\",\"kind\":\"scalar\",\"isList\":false,\"isRequired\":false,\"isUnique\":false,\"isId\":false,\"isReadOnly\":false,\"hasDefaultValue\":false,\"type\":\"String\",\"isGenerated\":false,\"isUpdatedAt\":false},{\"name\":\"admin\",\"kind\":\"scalar\",\"isList\":false,\"isRequired\":false,\"isUnique\":false,\"isId\":false,\"isReadOnly\":false,\"hasDefaultValue\":false,\"type\":\"Boolean\",\"isGenerated\":false,\"isUpdatedAt\":false},{\"name\":\"username\",\"kind\":\"scalar\",\"isList\":false,\"isRequired\":false,\"isUnique\":false,\"isId\":false,\"isReadOnly\":false,\"hasDefaultValue\":false,\"type\":\"String\",\"isGenerated\":false,\"isUpdatedAt\":false},{\"name\":\"password\",\"kind\":\"scalar\",\"isList\":false,\"isRequired\":false,\"isUnique\":false,\"isId\":false,\"isReadOnly\":false,\"hasDefaultValue\":false,\"type\":\"String\",\"isGenerated\":false,\"isUpdatedAt\":false},{\"name\":\"parking_pass_type\",\"kind\":\"scalar\",\"isList\":false,\"isRequired\":false,\"isUnique\":false,\"isId\":false,\"isReadOnly\":false,\"hasDefaultValue\":false,\"type\":\"String\",\"isGenerated\":false,\"isUpdatedAt\":false},{\"name\":\"address\",\"kind\":\"scalar\",\"isList\":false,\"isRequired\":false,\"isUnique\":false,\"isId\":false,\"isReadOnly\":false,\"hasDefaultValue\":false,\"type\":\"String\",\"isGenerated\":false,\"isUpdatedAt\":false},{\"name\":\"bio\",\"kind\":\"scalar\",\"isList\":false,\"isRequired\":false,\"isUnique\":false,\"isId\":false,\"isReadOnly\":false,\"hasDefaultValue\":false,\"type\":\"String\",\"isGenerated\":false,\"isUpdatedAt\":false},{\"name\":\"SESSIONS\",\"kind\":\"object\",\"isList\":true,\"isRequired\":true,\"isUnique\":false,\"isId\":false,\"isReadOnly\":false,\"hasDefaultValue\":false,\"type\":\"SESSIONS\",\"relationName\":\"SESSIONSToUSERS\",\"relationFromFields\":[],\"relationToFields\":[],\"isGenerated\":false,\"isUpdatedAt\":false}],\"primaryKey\":null,\"uniqueFields\":[],\"uniqueIndexes\":[],\"isGenerated\":false}},\"enums\":{},\"types\":{}}")
defineDmmfProperty(exports.Prisma, config.runtimeDataModel)


config.injectableEdgeEnv = () => ({
  parsed: {
    DATABASE_URL: typeof globalThis !== 'undefined' && globalThis['DATABASE_URL'] || typeof process !== 'undefined' && process.env && process.env.DATABASE_URL || undefined
  }
})

if (typeof globalThis !== 'undefined' && globalThis['DEBUG'] || typeof process !== 'undefined' && process.env && process.env.DEBUG || undefined) {
  Debug.enable(typeof globalThis !== 'undefined' && globalThis['DEBUG'] || typeof process !== 'undefined' && process.env && process.env.DEBUG || undefined)
}

const PrismaClient = getPrismaClient(config)
exports.PrismaClient = PrismaClient
Object.assign(exports, Prisma)

